# import torch
from teacher import Teacher
from model import CNN
from data import load_data, NoisyDataset, get_datasets
from utils import test_inference
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
        self.lr = 0.01  # og lr=0.01; lr between 0.0-1.0
        self.momentum = 0.0
        self.n_teachers = 5
        self.num_classes = 2
        self.sigma = 4  # noise scale (gaussian noise)

class KneeArguments:
    def __init__(self):
        self.batch_size = 300
        self.epochs = 10  # teacher epoch
        self.student_epochs = 30  # student epoch
        self.lr = 0.01  # og lr=0.01; lr between 0.0-1.0
        self.momentum = 0.0
        self.n_teachers = 5
        self.num_classes = 5
        self.sigma = 4  # noise scale (gaussian noise)

if __name__ == '__main__':
    # Note: (c = chest, k = knee)
    chest = True
    knee = False

    if chest:
        # === Train CNN model with DP-PATE using chest data ===
        print("Train CNN model with DP-PATE using chest data")

        # Load chest data
        print("Loading chest data")
        c_args = ChestArguments()
        train_teacher_dataset_c, train_student_dataset_c, test_dataset_c = get_datasets('chest')
        train_teacher_loader_c, train_student_loader_c, test_loader_c = load_data(train_teacher_dataset_c, train_student_dataset_c, test_dataset_c, c_args.batch_size)
        
        # Train teachers
        print("Training teacher")
        teacher = Teacher(c_args, CNN, c_args.num_classes, n_teachers=c_args.n_teachers)
        teacher.train(train_teacher_loader_c) #use valid dataset as a partition for teachers

        # Compile dataset from teachers
        print ("Creating Noisy Dataset")
        N = NoisyDataset(train_student_loader_c, teacher.predict) #use train dataset as the datset where labels are ignored (differentially private)
        
        # Train student model and get runtime, memory metrics
        print("Training Student")
        student = Student(c_args, CNN(c_args.num_classes))
        c_pate_runtime, c_pate_memory = get_memory_usage_and_runtime(student.train, (), {'dataset': N})

        # Get accuracy
        c_student_accuracy = test_inference(student, test_loader_c) * 100
        print ("Test Accuracy Student: ", c_student_accuracy , "%")

        # Get epsilon
        c_pate_epsilon = get_epsilon_momentents_gaussian_dp(len(train_student_dataset_c), c_args.sigma, c_args.student_epochs, c_args.batch_size)

        # Put metrics in txt file
        dump_metrics_to_json("dp_pate_chest_metrics.txt", c_pate_runtime, c_pate_memory, c_student_accuracy, c_pate_epsilon, is_dp=True)

    if knee:
        # === Train CNN model with DP-PATE using knee data ===
        print("\nTrain CNN model with DP-PATE using knee data")

        # Load knee data
        print("Loading knee data")
        k_args = KneeArguments()
        train_teacher_dataset_k, train_student_dataset_k, test_dataset_k = get_datasets('knee')
        train_teacher_loader_k, train_student_loader_k, test_loader_k = load_data(train_teacher_dataset_k, train_student_dataset_k,test_dataset_k, k_args.batch_size)

        # Train teachers
        print("Training teacher")
        teacher = Teacher(k_args, CNN, k_args.num_classes, n_teachers=k_args.n_teachers)
        teacher.train(train_teacher_loader_k) # use valid dataset as a partition for teachers

        # Compile dataset from teachers
        print ("Creating Noisy Dataset")
        N = NoisyDataset(train_student_loader_k, teacher.predict) #use train dataset as the datset where labels are ignored (differentially private)
        
        # Train student model and get runtime, memory metrics
        print("Training Student")
        student = Student(k_args, CNN(k_args.num_classes))
        k_pate_runtime, k_pate_memory = get_memory_usage_and_runtime(student.train, (), {'dataset': N})

        # Get accuracy
        k_student_accuracy = test_inference(student, test_loader_k) * 100
        print ("Test Accuracy Student: ", k_student_accuracy, "%")

        # Get epsilon
        k_pate_epsilon = get_epsilon_momentents_gaussian_dp(len(train_student_dataset_k), k_args.sigma, k_args.student_epochs, k_args.batch_size)

        # Put metrics in txt file
        dump_metrics_to_json("dp_pate_knee_metrics.txt", k_pate_runtime, k_pate_memory, k_student_accuracy, k_pate_epsilon, is_dp=True)
