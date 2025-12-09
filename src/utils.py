import torch.nn.utils.prune as tprune

def prune_model(model_params, amount):
    model = MLP()
    model.load_state_dict(model_params)
    # Pruning
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tprune.l1_unstructured(module, name='weight', amount=amount)

    #Remove the pruning reparametrizations to make the model explicitly sparse
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tprune.remove(module, 'weight')
    return model.state_dict()


def check_sparsity(state_dict, verbose=False):
    total_zeros = 0
    total_params = 0

    for name, tensor in state_dict.items():

        num_params = tensor.numel()
        num_zeros = torch.sum(tensor == 0).item()

        total_params += num_params
        total_zeros += num_zeros

        if verbose:
            layer_sparsity = (num_zeros / num_params) * 100
            print(f"Layer: {name} | Sparsity: {layer_sparsity:.2f}%")

    if total_params == 0:
        return 0.0

    global_sparsity = (total_zeros / total_params) * 100
    return global_sparsity