import torch
import torchvision.models as models
from tqdm import tqdm
from time import time


def scan_batch_size(model, resolutions, max_batch_size=128, device="cuda:0"):
    """
    Scans the max batch size that can be supported for all specified resolutions
    :param model: model to be used for benchmarking
    :param resolutions: list of resolutions to be used for benchmarking
    :param max_batch_size: maximum size of batch size to be tested
    :param device: device to use for the tests
    :return: max supported batch size per resolution and memory footprint per sample (in MB)
    """

    def _infer_max_batch_size(model, resolution):
        model.to(device)
        batch_size = 0
        avg_memory = 0
        print("Searching for max batch size on " + device + " for input of size", resolution)
        try:
            dummy_var = 0
            for cur_batch in tqdm(range(1, max_batch_size)):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                # Run inference a few time to stabilize memory usage
                for j in range(3):
                    input_tensor = torch.randn((cur_batch, 3, resolution[0], resolution[1]))
                    input_tensor = input_tensor.to(device)
                    results = model(input_tensor)
                    dummy_var += torch.sum(results.cpu()).detach() * 1e-5
                avg_memory += (torch.cuda.memory_stats(device)['allocated_bytes.all.peak'] / cur_batch)
                torch.cuda.synchronize()

        except RuntimeError:
            # This is the maximum size that we can handle
            batch_size = cur_batch - 1
            if batch_size != 0:
                avg_memory = avg_memory / batch_size
        # Convert memory to MB
        avg_memory = avg_memory / (1024 * 1024)
        return batch_size, avg_memory

    batch_size_per_resolution = []
    memory_per_resolution = []
    for resolution in resolutions:
        batch, mem = _infer_max_batch_size(model, resolution)
        batch_size_per_resolution.append(batch)
        memory_per_resolution.append(mem)
    return batch_size_per_resolution, memory_per_resolution


def benchmark_speed(model, batch_sizes, resolutions, n_repeats=20, device='cuda:0'):
    """
    Measures inference speed for a model across all batch sizes and resolutions
    :param model:  model to be used for benchmarking
    :param batch_sizes: batch sizes to be used for different resolutions
    :param resolutions: resolutions to be used for benchmarking
    :param n_repeats: number of repeated runs
    :param device: device to use for the tests
    :return: average speed (FPS) per resolution
    """

    assert len(batch_sizes) == len(resolutions)
    speed_per_resolution = []
    model.to(device)

    for batch, resolution in tqdm(zip(batch_sizes, resolutions)):
        if batch > 0:
            print("Benchmarking for ", resolution)
            total_time = 0
            dummy_var = 0
            for i in range(n_repeats):
                torch.cuda.empty_cache()
                input_tensor = torch.randn((batch, 3, resolution[0], resolution[1]))
                start_time = time()
                input_tensor = input_tensor.to(device)
                results = model(input_tensor)
                dummy_var += torch.sum(results.cpu()).detach() * 1e-5
                torch.cuda.synchronize()
                elapsed_time = time() - start_time
                total_time += elapsed_time
            fps = batch / (total_time / n_repeats)
            speed_per_resolution.append(fps)
        else:
            speed_per_resolution.append(0)
    return speed_per_resolution


if __name__ == '__main__':
    resolutions = ((360, 640), (720, 1280), (1080, 1920))

    model = models.resnet18()
    batch_sizes, memory = scan_batch_size(model, resolutions, device='cuda:0')
    print("Supported batch sizes = ", batch_sizes)
    print("Memory usage per resolution = ", memory)

    # supported_batch_sizes = [49, 12, 5]
    # fps = benchmark_speed(model, supported_batch_sizes, resolutions, device='cuda:0')
    # print("Speed (FPS) usage per resolution = ", fps)
