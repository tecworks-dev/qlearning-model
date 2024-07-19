import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import optuna
from optuna.trial import TrialState
from torch.utils.tensorboard import SummaryWriter
from model import SwinTransformer
from agent import QLearningAgent
from data import get_dataloaders


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']


def objective(trial, config, trainloader, valloader, checkpoint_path, device):
    model = SwinTransformer(config['model']).to(device)
    agent = QLearningAgent(model, config, trainloader)

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, agent.optimizer, agent.scheduler, checkpoint_path)
        print(f"Loaded checkpoint from epoch {start_epoch}")

    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        for images, _ in trainloader:
            images = images.to(device)
            state = images
            action = torch.randint(0, config['model']['num_classes'], (images.size(0),)).to(device)
            reward = torch.ones(images.size(0),).to(device)
            next_state = images
            done = torch.zeros(images.size(0),).to(device)

            agent.update(state, action, reward, next_state, done)

        save_checkpoint(model, agent.optimizer, agent.scheduler, epoch, checkpoint_path)
        print(f"Epoch {epoch + 1} completed and checkpoint saved")

        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch + 1}: Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

        trial.report(accuracy, epoch)
        writer.add_scalar('Validation Accuracy', accuracy, epoch)
        writer.add_scalar('Validation Precision', precision, epoch)
        writer.add_scalar('Validation Recall', recall, epoch)
        writer.add_scalar('Validation F1-Score', f1, epoch)

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epochs_no_improve >= config['train']['patience']:
            print(f"Early stopping after {epoch + 1} epochs due to no improvement in validation accuracy.")
            break

    return best_val_acc


def train_model(config, device):
    trainloader, valloader, testloader = get_dataloaders(config)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, config, trainloader, valloader, 'swin_transformer_checkpoint.pth', device), n_trials=config['train']['trials'], timeout=config['train']['timeout'])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial.params


def generate(model, data_loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            with autocast():
                output = model(images)
                outputs.append(output)
    return torch.cat(outputs)


def evaluate_model(config, device):
    _, _, testloader = get_dataloaders(config)

    model = SwinTransformer(config['model']).to(device)
    agent = QLearningAgent(model, config, testloader)

    if os.path.exists('swin_transformer_checkpoint.pth'):
        load_checkpoint(model, agent.optimizer, agent.scheduler, 'swin_transformer_checkpoint.pth')
        print(f"Loaded checkpoint for evaluation")

    output = generate(model, testloader, device)
    print("Generated output shape:", output.shape)

    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    writer.add_scalar('Test Accuracy', accuracy)
    writer.add_scalar('Test Precision', precision)
    writer.add_scalar('Test Recall', recall)
    writer.add_scalar('Test F1-Score', f1)
    writer.close()
