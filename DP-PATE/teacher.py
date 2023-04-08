import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


class Teacher:
    """Implementation of teacher models.
       Teacher models are ensemble of models which learns directly disjoint splits of the sensitive data
       The ensemble of teachers are further used to label unlabelled public data on which the student is
       trained.
       Args:
           n_teachers (int): Number of teachers
           model (CNN): CNN model class
           models [dict]: dictionary of all the teacher models
           args (Arguments object): An object of Arguments class with required hyperparameters
           num_classes (int): the number of classes for the model
           stdev (int): scale for the Normal (Gaussian) distribution
    """

    def __init__(self, args, model, num_classes, n_teachers=1, stdev=1):

        self.n_teachers = n_teachers
        self.model = model
        self.models = {}
        self.args = args
        self.num_classes = num_classes
        self.init_models()
        self.sigma = stdev  # Gaussian Noise

    def init_models(self):
        """Initialize teacher models according to number of required teachers"""

        name = "model_"
        for index in range(0, self.n_teachers):

            model = self.model(self.num_classes)
            self.models[name + str(index)] = model

    def addnoise(self, x):
        """Adds Gaussian noise to histogram of counts
           Args:
                x (torch tensor): histogram of counts
           Returns:
                count (torch tensor): Noisy histogram of counts
        """

        m = Normal(torch.tensor([0.0]), torch.tensor([self.sigma]))
        count = x + m.sample()

        return count

    def split(self, dataset):
        """Function to split the dataset into non-overlapping subsets of the data
           Args:
               dataset (torch tensor): The dataset in the form of (image,label)
           Returns:
               split: Split of dataset
        """

        ratio = 1
        iters = 0
        index = 0
        split = []
        last_batch = ratio * self.n_teachers

        for teacher in range(0, self.n_teachers):

            split.append([])

        for (data, target) in dataset:
            if iters % ratio == 0 and iters != 0:

                index += 1

            split[index].append([data, target])
            iters += 1

            if iters == last_batch:
                return split

        return split

    def train(self, dataset):
        """Function to train all teacher models.
           Args:
                dataset (torch tensor): Dataset used to train teachers in format (image,label)
        """

        split = self.split(dataset)

        for epoch in range(1, self.args.epochs + 1):

            index = 0
            for model_name in self.models:

                print("TRAINING ", model_name)
                print("EPOCH: ", epoch)
                self.loop_body(split[index], model_name)
                index += 1

    def loop_body(self, split, model_name):
        """Body of the training loop.
           Args:
               split: Split of the dataset which the model has to train.
               model_name: Name of the model.
        """

        model = self.models[model_name]
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        iters = 0
        loss = 0.0
        for (data, target) in split:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            iters += 1
        # Print loss by making using of log intervals
        print("Loss")
        if isinstance(loss, float):
            print(loss)
        else:
            print(loss.item())

    def aggregate(self, model_votes, batch_size):
        """Aggregate model output into a single tensor of votes of all models.
           Args:
                model_votes: Model output
                batch_size: Number of datapoints
           Returns:
                counts: Torch tensor with counts across all models
                model_counts: Torch tensor with counts of each model
           """

        counts = torch.zeros([batch_size, self.args.num_classes])
        model_counts = torch.zeros([self.args.n_teachers, batch_size])
        model_index = 0

        for model in model_votes:

            index = 0

            for tensor in model_votes[model]:
                for val in tensor:

                    counts[index][val] += 1
                    model_counts[model_index][index] = val
                    index += 1

            model_index += 1

        return counts, model_counts

    def save_models(self):
        no = 0
        for model in self.models:

            torch.save(self.models[model].state_dict(), "models/" + model)
            no += 1

        print("\n")
        print("MODELS SAVED")
        print("\n")

    def load_models(self):

        path_name = "model_"

        for i in range(0, self.args.n_teachers):

            model_a = self.model()
            self.models[path_name + str(i)] = torch.load("models/" + path_name + str(i))
            self.models[path_name + str(i)] = model_a.load_state_dict()

    def predict(self, data):
        """Make predictions using Noisy-max using Gaussian mechanism.
           Args:
                data: Data for which predictions are to be made
           Returns:
                output: Predictions for the data
        """

        model_predictions = {}

        for model in self.models:

            out = []
            output = self.models[model](data)
            output = output.max(dim=1)[1]
            out.append(output)

            model_predictions[model] = out

        counts, model_counts = self.aggregate(model_predictions, len(data))
        counts = counts.apply_(self.addnoise)

        predictions = []

        for batch in counts:
            predictions.append(batch.max(dim=0)[1].long().clone().detach())

        output = {"predictions": predictions, "counts": counts, "model_counts": model_counts}

        return output
