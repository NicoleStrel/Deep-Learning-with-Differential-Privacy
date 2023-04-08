import torch


def test_inference(student, testloader):
    """
    Returns the test accuracy
    """
    total, correct = 0.0, 0.0

    for data, labels in testloader:
        predictions = student.predict_v2(data)
        _, pred_labels = torch.max(predictions, 1)

        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy
