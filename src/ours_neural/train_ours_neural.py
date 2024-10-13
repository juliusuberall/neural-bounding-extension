import torch.optim as optim
import torch
import matplotlib.pyplot as plt

from src.loss.loss import BCELossWithClassWeights
from src.metrics.helper import print_metrics
from src.metrics.metrics_calculator import MetricsCalculator
from src.wiring import get_source_data, get_training_data, get_model


def train_ours_neural(object_name, query, dimension, metrics_registry):
    print(f"oursNeural {object_name} {dimension}D {query} query")

    # hyperparameters
    n_regions = 50_000
    n_samples = 1500 if dimension == 4 else 500

    # load data
    data = get_source_data(object_name=object_name, dimension=dimension)

    # initialise model
    model = get_model(query=query, dimension=dimension)

    # initialise asymmetric binary cross-entropy loss function, and optimiser
    class_weight = 1
    criterion = BCELossWithClassWeights(positive_class_weight=1, negative_class_weight=1)
    optimiser = optim.Adam(model.parameters(), lr=0.0001)

    # initialise counter and print_frequency
    weight_schedule_frequency = 250_000
    total_iterations = weight_schedule_frequency * 200  # set high iterations for early stopping to terminate training
    evaluation_frequency = weight_schedule_frequency // 5
    print_frequency = 1000  # print loss every 1k iterations

    # instantiate count for early stopping
    count = 0

    # initialise plot data lists for loss on each exit
    plt_iteration, plt_loss1, plt_loss2, plt_loss3, plt_lossMean = [],[],[],[],[]

    for iteration in range(total_iterations):
        features, targets = get_training_data(data=data, query=query, dimension=dimension, n_regions=n_regions,
                                              n_samples=n_samples)

        # forward pass
        output = model(features)

        # compute average loss for all exits
        loss1 = criterion(output[0], targets)
        loss2 = criterion(output[1], targets)
        loss3 = criterion(output[2], targets)
        l1_weight = 1.0
        l2_weight = 1.0
        l3_weight = 1.0
        loss = (loss1*l1_weight + loss2*l2_weight + loss3*l3_weight) / (l1_weight+l2_weight+l3_weight)

        # zero gradients, backward pass, optimiser step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # print loss 
        if (iteration + 1) % print_frequency == 0 or iteration == 0:
            print(f'Iteration: {iteration + 1}, Loss: {loss.item()},')
            # torch.save(model, 'prototyping/models/model_' + str(iteration + 1) + '.pth')
            # print("Model saved")

            # create loss plot
            plt_iteration.append(iteration + 1)
            plt_loss1.append(loss1.detach().numpy())
            plt_loss2.append(loss2.detach().numpy())
            plt_loss3.append(loss3.detach().numpy())
            plt_lossMean.append(loss.detach().numpy())
            plt.figure(figsize=(8, 6))
            plt.plot(plt_iteration, plt_loss1, label=f'L1 - {l1_weight}', color='blue')
            plt.plot(plt_iteration, plt_loss2, label=f'L2 - {l2_weight}', color='orange')
            plt.plot(plt_iteration, plt_loss3, label=f'L3 - {l3_weight}', color='pink')
            plt.plot(plt_iteration, plt_lossMean, label='Mean', color='black', linestyle='--')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Layer Loss History')
            plt.legend()
            plt.savefig("prototyping/inferenceAnalysisResults/LayerLossHistory.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved new layer loss history plot")


        if (iteration + 1) % evaluation_frequency == 0 or iteration == 0:
            prediction = (loss.cpu().detach() >= 0.5).float().numpy()
            target = targets.cpu().detach().numpy()
            metrics = MetricsCalculator.calculate(prediction=prediction, target=target)
            print_metrics(metrics)

        if (iteration + 1) % evaluation_frequency == 0:
            prediction = (loss.cpu().detach() >= 0.5).float().numpy()
            target = targets.cpu().detach().numpy()
            metrics = MetricsCalculator.calculate(prediction=prediction, target=target)

            # if convergence to FN 0 is not stable yet and still oscillating
            # let the model continue training
            # by resetting the count
            if count != 0 and metrics["false negatives"] != 0.:
                count = 0

            # ensure that convergence to FN 0 is stable at a sufficiently large class weight
            if metrics["false negatives"] == 0.:
                count += 1

            if count == 3:
                # save final training results
                metrics_registry.metrics_registry["oursNeural"] = {
                    "class weight": class_weight,
                    "iteration": iteration+1,
                    "false negatives": metrics["false negatives"],
                    "false positives": metrics["false positives"],
                    "true values": metrics["true values"],
                    "total samples": metrics["total samples"],
                    "loss": f"{loss:.5f}"
                }

                # early stopping
                print("early stopping\n")
                break

        # schedule increases class weight by 20 every 500k iterations
        if (iteration + 1) % weight_schedule_frequency == 0 or iteration == 0:
            if iteration == 0:
                pass
            elif (iteration + 1) == weight_schedule_frequency:
                class_weight = 20
            else:
                class_weight += 20

            criterion.negative_class_weight = 1.0 / class_weight

            print("class weight", class_weight)
            print("BCE loss negative class weight", criterion.negative_class_weight)

