import torch
from teacher import Teacher
from model import CNN
from data import load_data, NoisyDataset, get_datasets
from utils import accuracy, split
from student import Student
from metrics_calc_helper_functions import get_memory_usage_and_runtime, get_epsilon_momentents_gaussian_dp, \
    dump_metrics_to_json

# chest batch_size = 100
# chest num_classes = 2
# knee batch_size = 300
# knee num_classes = 3
# for all: epochs = up to you


class ChestArguments:
    def __init__(self):
        self.batch_size = 100
        self.epochs = 10  # epoch for teacher
        self.student_epochs = 30  # epoch for students
        self.lr = 0.005  # og lr=0.01; lr between 0.0-1.0
        self.momentum = 0.0
        self.no_cuda = False  # does not seem to be used
        self.seed = 1  # does not seem to be used
        self.log_interval = 30  # does not seem to be used
        self.n_teachers = 50
        self.save_model = False  # does not seem to be used

        self.num_classes = 2
        self.sigma = 4  # noise scale (gaussian noise)


class KneeArguments:
    def __init__(self):
        self.batch_size = 300
        self.epochs = 10  # teacher epoch
        self.student_epochs = 30  # student epoch
        self.lr = 0.005  # og lr=0.01; lr between 0.0-1.0
        self.momentum = 0.0
        self.no_cuda = False  # does not seem to be used
        self.seed = 1  # does not seem to be used
        self.log_interval = 30  # does not seem to be used
        self.n_teachers = 50
        self.save_model = False  # does not seem to be used

        self.num_classes = 5
        self.sigma = 4  # noise scale (gaussian noise)


if __name__ == '__main__':
    # Note: (c = chest, k = knee)

    # === Get chest and knee data ===

    # chest data
    batch_size_c = 100
    train_dataset_c, valid_dataset_c, test_dataset_c = get_datasets('chest')
    train_loader_c, valid_loader_c, test_loader_c = load_data(train_dataset_c, valid_dataset_c,
                                                              test_dataset_c, batch_size_c)

    # knee data
    batch_size_k = 300
    train_dataset_k, valid_dataset_k, test_dataset_k = get_datasets('knee')
    train_loader_k, valid_loader_k, test_loader_k = load_data(train_dataset_k, valid_dataset_k,
                                                              test_dataset_k, batch_size_k)

    # === Train CNN model with DP-PATE using chest data ===
    print("Train CNN model with DP-PATE using chest data")
    c_args = ChestArguments()

    # Train teachers
    teacher = Teacher(c_args, CNN, c_args.num_classes, n_teachers=c_args.n_teachers)
    teacher.train(train_loader_c)

    # Teacher accuracy
    teacher_targets = []
    predict = []

    counts = []
    original_targets = []

    for data, target in test_loader_c:
        output = teacher.predict(data)

        arr_target = []
        teacher_targets.append(target)
        original_targets.append(target)
        predict.append(output["predictions"])
        counts.append(output["model_counts"])

    # print("Accuracy: ", accuracy(torch.tensor(predict), teacher_targets))

    print("\n")
    print("\n")

    # Training students
    print("Training Student")

    print("\n")
    print("\n")

    # train model and get runtime, memory metrics
    student = Student(c_args, CNN(c_args.num_classes))
    N = NoisyDataset(test_loader_c, teacher.predict)
    c_pate_runtime, c_pate_memory = get_memory_usage_and_runtime(student.train, (), {'dataset': N})

    results = []
    targets = []

    total = 0.0
    correct = 0.0

    for data, target in valid_loader_c:
        predict_lol = student.predict(data)
        correct += float((predict_lol == (target)).sum().item())
        total += float(target.size(0))

    c_student_accuracy = (correct / total) * 100
    print("Private Baseline: ", c_student_accuracy)

    # get epsilon
    c_pate_epsilon = get_epsilon_momentents_gaussian_dp(len(test_dataset_c), c_args.sigma,
                                                        c_args.student_epochs, c_args.batch_size)

    # put metrics in txt file
    dump_metrics_to_json("dp_pate_chest_metrics.txt", c_pate_runtime, c_pate_memory,
                         c_student_accuracy, c_pate_epsilon, is_dp=True)

    # === Train CNN model with DP-PATE using knee data ===
    print("\nTrain CNN model with DP-PATE using knee data")
    k_args = KneeArguments()

    # Train teachers
    teacher = Teacher(k_args, CNN, k_args.num_classes, n_teachers=k_args.n_teachers)
    teacher.train(train_loader_k)

    # Teacher accuracy
    teacher_targets = []
    predict = []

    counts = []
    original_targets = []

    for data, target in test_loader_k:
        output = teacher.predict(data)

        arr_target = []
        teacher_targets.append(target)
        original_targets.append(target)
        predict.append(output["predictions"])
        counts.append(output["model_counts"])

    # print("Accuracy: ", accuracy(torch.tensor(predict), teacher_targets))

    print("\n")
    print("\n")

    # Training students
    print("Training Student")

    print("\n")
    print("\n")

    student = Student(k_args, CNN(k_args.num_classes))
    N = NoisyDataset(test_loader_k, teacher.predict)
    k_pate_runtime, k_pate_memory = get_memory_usage_and_runtime(student.train, (), {'dataset': N})

    results = []
    targets = []

    total = 0.0
    correct = 0.0

    for data, target in valid_loader_k:
        predict_lol = student.predict(data)
        correct += float((predict_lol == (target)).sum().item())
        total += float(target.size(0))

    k_student_accuracy = (correct / total) * 100
    print("Private Baseline: ", k_student_accuracy)

    # get epsilon
    k_pate_epsilon = get_epsilon_momentents_gaussian_dp(len(test_dataset_k), k_args.sigma,
                                                        k_args.student_epochs, k_args.batch_size)

    # put metrics in txt file
    dump_metrics_to_json("dp_pate_knee_metrics.txt", k_pate_runtime, k_pate_memory,
                         k_student_accuracy, k_pate_epsilon, is_dp=True)

