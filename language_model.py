import sys
import logging
import torch
import speechbrain as sb
from datasets import load_dataset
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger("Language model logger")


class LM(sb.core.Brain):

    hparams_file, run_opts, overrides = sb.parse_arguments(["language_model.yaml"])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)


    def compute_forward(self, batch, stage):
        """
        Predicts next word given previous ones.

        Arguments:
        --------
        batch: paddedbatch
                This batch object contains all the relevant tensors for computation.
        stage: sb.stage
                One for sb.stage.Train, sb.stage.Valid, or sb.stage.Test

        Returns:
        -------
        predictions: torch.Tensor
            A tensor containing the posterior probabilities (predictions)
        """
        batch = batch.to(self.device)
        tokens_bos, _ = batch.tokens_bos
        pred = self.hparams["model"](tokens_bos)
        return pred

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the loss given the predicted and targeted outputs.

        Arguments:
        -------
        prediction: torch.Tensor
                The posterior probabilities from 'compute_forward'
        batch: PaddedBatch
                This batch object contains all the relevant tensors for computation.
        stage: sb.Stage
                One of sb.Stage.Train, sb.Stage.Valid, or sb.Stage.Test.

        Returns:
        ------
        loss: torch.Tensor
                A one-element tensor used for backpropagating the gradient.
        """
        batch = batch.to(self.device)
        tokens_eos, tokens_len = batch.tokens_eos
        loss = self.hparams["compute_cost"](
            predictions, tokens_eos, length=tokens_len
        )
        return loss

    def fit_batch(self, batch):
        """
        Runs all the steps needed to train the model on a single batch.

        Arguments:
        -------
        batch: PaddedBatch
                this batch object contains all the relevant tensors for computation.
        Returns:
        ------
        Loss: torch.Tensor
                A tensor containing the loss (single read number)
        """
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # loss backpropagation (gradient computation)
        (loss / self.hparams["accu_steps"]).backward()

        # Manage gradient accumulation
        if self.step % self.hparams["accu_steps"] == 0:

            # Gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            # update the parameters
            self.optimizer.step()

            # Reset the gradient
            self.optimizer.zero_grad()

            if isinstance(
                self.hparams["lr_annealing"], sb.nnet.schedulers.NoamScheduler
            ) or isinstance(
                self.hparams["lr_annealing"], sb.nnet.schedulers.CyclicCosineScheduler
            ):
                self.hparams["lr_annealing"](self.optimizer)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """
        Gets called at the end of an epoch

        Arguments:
        --------
        stage: sb.Stage
                One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST
        stage_loss: float
                The average loss of all the data processed in this stage.
        epoch: int
                The currently starting epoch. This is passed 'None' during the test stage
        """

        # store the train loss untill the validation stage.
        if stage== sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # summarize the statistics from the stage for record-keeping
        else:
            stats = {
                "loss" : stage_loss,
            }

        # At the end of validation we can wrote
        if stage == sb.Stage.VALID:

            # update the learning rate
            old_lr, new_lr = self.hparams["lr_annealing"](stage_loss)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger write a summary to stdout and to the logfile
            self.hparams["train_logger"].log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])

        # We also write statistics about test data to stdout and to the logfile
        if stage == sb.Stage.TEST:

            self.hparams["train_logger"].log_stats(
                {"Epoch loaded": self.hparams["epoch_counter"].current},
                test_stats=stats,
            )

    def dataio_prepare(self, hparams):
        """
        This function prepares the dataset to be used in the brain class
        It also defines the data processing pipeline through user-defined functions

        The language model is trained with text files specified by the user in the hyperparameter file.

        Arguments:
        --------
        hparams: dict
                This dictionary is loaded from the language_model.yaml file, and it includes all the
                hyperparameters needed for dataset construction and loading.

        Returns:
        ------
        datasets: list
                List containing "train", "valid", and "test" sets that corresponds to the appropriate
                DynamicItemDataset object.
        """

        logging.info("generating datasets...")

        # Prepare datasets
        datasets = load_dataset(
            "text",
            data_files={
                "train": hparams["lm_train_data"],
                "valid": hparams["lm_valid_data"],
                "test": hparams["lm_test_data"]
            },
        )

        # Convert the huggingface's dataset to DynamicItemDataset
        train_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
            datasets["train"]
        )
        valid_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
            datasets["valid"]
        )
        test_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
            datasets["test"]
        )

        datasets = [train_data, valid_data, test_data]
        tokenizer = hparams["tokenizer"]

        # Define text processing pipeline. we start from the raw text and then
        # encode it using the tokenizer. The tokens with bos are used for feeding
        # the neural network, the tokens with eos for computing the cost function.
        @sb.utils.data_pipeline.takes("text")
        @sb.utils.data_pipeline.provides("text", "tokens_bos", "tokens_eos")
        def text_pipeline(text):
            yield text
            tokens_list = tokenizer.encode_as_ids(text)
            tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
            yield tokens_bos
            tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
            yield tokens_eos

        sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

        # Set outputs to add into the batch. The batch variable will contain
        # all these fields (e.g, batch.id, batch.text, batch.tokens.bos,..)
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "text", "tokens_bos", "tokens_eos"]
        )

        return train_data, valid_data, test_data


    def run(self):
        """
        This function runs the previous function. The main function
        """

        # Create experiment folder
        sb.create_experiment_directory(
            experiment_directory=self.hparams["output_folder"],
            hyperparams_to_save=self.hparams_file,
            overrides=self.overrides,
        )

        # Create the dataset objects "train", "valid" and "test"
        train_data, valid_data, test_data = self.dataio_prepare(self.parsed_yaml_file)



        # initialize the Brain object to prepare for LM training
        lm_brain = LM(
            modules=self.parsed_yaml_file["modules"],
            opt_class=self.parsed_yaml_file["optimizer"],
            hparams=self.hparams,
            run_opts=self.run_opts,
            checkpointer= self.hparams["checkpointer"],

        )

        # The fit() method iterates the training loop, calling the methods necessary to update
        # the parameters of the model. Since all objects with changing state are managed by
        # the checkpointer, training can be stopped at any point, and will be resumed on the next call
        lm_brain.fit(
            lm_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=self.hparams["train_dataloader_opts"],
            valid_loader_kwargs=self.hparams["valid_dataloader_opts"],
        )

        # load best checkpoint for evaluation
        test_stats = lm_brain.evaluate(
            test_data,
            min_key="loss",
            test_loader_kwargs=self.hparams["test_dataloader_opts"],
        )