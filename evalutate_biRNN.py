import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score
from data import getBatch


def evaluate_predictions(tag_truths, tag_predictions, intent_truths, intent_predictions):
    # flatten all the batches to 1d array
    flattened_tag_truths = torch.flatten(torch.stack(tag_truths))
    flattened_tag_predictions = torch.flatten(torch.stack(tag_predictions))

    # <PAD> has index 0, we should not include padding in the score
    zero_tags = flattened_tag_truths == 0
    f1_tag_score = f1_score(flattened_tag_truths[(~zero_tags)], flattened_tag_predictions[(~zero_tags)],
                            average='micro')

    intent_accuracy = accuracy_score(torch.flatten(torch.stack(intent_truths)), torch.flatten(torch.stack(intent_predictions)))
    return f1_tag_score, intent_accuracy


def evaluate(biRNN, data, batch_size): 
    use_cuda = torch.cuda.is_available()

    biRNN.eval()

    tag_truths = []
    tag_predictions = []

    intent_truths = []
    intent_predictions = []

    loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
    loss_function_2 = nn.CrossEntropyLoss()

    losses = []

    with torch.no_grad():
        for batch in getBatch(batch_size, data):
            x, y_1, y_2 = zip(*batch)
            x = torch.cat(x).cuda() if use_cuda else torch.cat(x)
            tag_target = torch.cat(y_1).cuda() if use_cuda else torch.cat(y_1)
            tag_target = tag_target.view(-1)

            intent_target = torch.cat(y_2).cuda() if use_cuda else torch.cat(y_2)
            x_mask = torch.cat([torch.tensor(tuple(map(lambda s: s == 0, t.data)), dtype=torch.bool) for t in x])
            x_mask = x_mask.view(batch_size, -1)

            tag_score, intent_score = biRNN(x, x_mask)

            loss_1 = loss_function_1(tag_score,tag_target.view(-1))
            loss_2 = loss_function_2(intent_score,intent_target)

            loss = loss_1+loss_2
            losses.append(loss.data.cpu().numpy().item()) 

            # get predicted tags
            _, predicted_tag_indices = torch.max(tag_score, 1)
            tag_truths.append(tag_target.cpu())
            tag_predictions.append(predicted_tag_indices.cpu())

            # get predicted intent
            _, predicted_intent_indices = torch.max(intent_score, 1)
            intent_truths.append(intent_target.cpu())
            intent_predictions.append(predicted_intent_indices.cpu())

    f1_tag_score, intent_accuracy = evaluate_predictions(tag_truths, tag_predictions, intent_truths, intent_predictions)
    biRNN.train()
    return losses, f1_tag_score, intent_accuracy
