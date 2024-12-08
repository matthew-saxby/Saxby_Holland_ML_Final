def printPerformance(train_accuracy, train_f1, val_accuracy, val_f1):
    print('---- Training performance----')
    print('Train Accuracy:', train_accuracy)
    print('Train F1 Score', train_f1)
    print('----Validation performance----')
    print('Validation Accuracy:', val_accuracy)
    print('Validation F1 Score', val_f1)


def printTestPerformance(train_accuracy, train_f1, test_accuracy, test_f1):
    print()
    print('----Testing performance----')
    print('Testing Accuracy:', test_accuracy)
    print('Testing F1 Score', test_f1)