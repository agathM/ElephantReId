import torch

def main():
    print("=== PyTorch Test ===")
    
    # Version
    print("Torch version:", torch.__version__)

    # CPU test
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = x + y

    print("\nCPU test OK")
    print("Tensor result:\n", z)

    # CUDA test
    print("\nCUDA available:", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
    torch.cuda.get_arch_list()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        z_gpu = x_gpu + y_gpu

        print("GPU test OK")
        print("Device:", device)
        print("GPU result:\n", z_gpu)

        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("GPU not available (CPU only)")

    # Gradient test
    print("\nGradient test")
    x = torch.randn(3, 3, requires_grad=True)
    y = x * 2
    y.sum().backward()

    print("Grad OK:", x.grad is not None)
    print("Gradient:\n", x.grad)

    print("\n=== ALL TESTS DONE ===")


if __name__ == "__main__":
    main()