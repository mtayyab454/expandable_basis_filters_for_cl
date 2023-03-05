import torch
from cifar.models.resnet import resnet32_multitask, resnet32
from cifar.models.testnet import testcnn_multitask, testcnn

from multitask_helpers import trace_model, get_basis_channels_from_t, display_stats

if __name__ == '__main__':
    # Create a model and train on task 1
    model = resnet32(10)
    model.eval()
    input_tensor = torch.randn(1, 3, 32, 32)
    model_output = model(input_tensor)

    # Get optimal number of filters for each conv2d layer in the model trained on task 1
    num_conv, num_linear, in_channels, out_channels, basis_channels, layer_type = trace_model(model)
    _, _, basis_channels = get_basis_channels_from_t(model, [1.0]*num_conv)

    # Create a multitask model with the basis channels estimated above
    basis_model = resnet32_multitask(basis_channels, [False]*len(basis_channels), [10])

    # Initilize the task 1 parameters of multitask model using the weights of conv2d model
    basis_model.load_t1_weights(model)
    basis_model.eval()

    basis_model_output = basis_model(input_tensor)

    # Check if the results are same.
    print( (model_output-basis_model_output).abs().sum() )
    print(torch.allclose(model_output, basis_model_output, atol=1e-2))  # True



