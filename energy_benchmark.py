import subprocess
import time
from time import sleep
import threading
import torch
from tqdm import tqdm
import torchvision.models as models


def get_instant_power(gpu_id):
    """
    This function decodes the nvidia-smi output provided by Driver Version: 510.54 in order to get instant power for
    a specific gpu_id
    :param gpu_id: gpu id that we are interested in
    :return:
    """

    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    output = str(result.stdout)
    output = output.split('\\n')
    line = str(output[9 + 4 * gpu_id])
    line = ' '.join(line.split()).replace('|', '').split(' ')
    current_power = float(line[4][:-1])
    return current_power


def estimate_fn_power(fn, sampling_interval=0.1, gpu_id=0):
    start = time.time()
    total_energy = 0
    th = threading.Thread(target=fn)
    th.start()

    while True:
        sleep(sampling_interval)
        poll_start_time = time.time()
        if th.is_alive():
            current_power = get_instant_power(gpu_id)
            poll_time = time.time() - poll_start_time
            total_energy += current_power * (sampling_interval + poll_time)
        else:
            break
    return total_energy


def fn():
    time.sleep(4)


def benchmark_energy(model, resolutions, batch_sizes, n_repeats=20, gpu_id=0):
    device = 'cuda:' + str(gpu_id)
    assert len(batch_sizes) == len(resolutions)
    energy_per_resolution = []
    model.to(device)

    for batch, resolution in tqdm(zip(batch_sizes, resolutions)):
        if batch > 0:
            print("Benchmarking for ", resolution)
            total_energy = 0

            for i in range(n_repeats):
                torch.cuda.empty_cache()
                input_tensor = torch.randn((batch, 3, resolution[0], resolution[1]))
                input_tensor = input_tensor.to(device)

                def bench_fn():
                    results = model(input_tensor)
                    res = torch.sum(results.cpu()).detach() * 1e-5
                    torch.cuda.synchronize()

                total_energy += estimate_fn_power(bench_fn, sampling_interval=0.01, gpu_id=gpu_id)
            total_energy = total_energy / n_repeats
            total_energy = total_energy / batch
            energy_per_resolution.append(total_energy)
        else:
            energy_per_resolution.append(0)
    return energy_per_resolution

if __name__ == '__main__':

    resolutions = ((360, 640), (720, 1280), (1080, 1920))
    model = models.resnet18()
    supported_batch_sizes = [49, 12, 5]
    results = benchmark_energy(model, resolutions, supported_batch_sizes)
    print("Energy usage (J) per resolution: ", results)
