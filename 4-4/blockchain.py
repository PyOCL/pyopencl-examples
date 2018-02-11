import os
import math
import numpy
import pyopencl as cl
import pyopencl.array
import time

if __name__ == '__main__':

    print('load program from cl source file')
    f = open('blockchain.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    print('prepare data ... ')
    start_time = time.time()
    # prepare host memory for OpenCL
    data_str = 'abc'
    input_data = numpy.fromstring(data_str, dtype=numpy.uint8)
    output_data = numpy.zeros(32, dtype=numpy.uint8)
    nonce = numpy.zeros(1, dtype=numpy.uint32)
    time_hostdata_loaded = time.time()

    print('create context ...')
    ctx = cl.create_some_context()
    print('create command queue ...')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    time_ctx_queue_creation = time.time()

    # prepare device memory for OpenCL
    print('prepare device memory for input / output')
    cl_input_data = cl.array.to_device(queue, input_data)
    cl_output_data = cl.array.to_device(queue, output_data)
    cl_nonce = cl.array.to_device(queue, nonce)
    time_devicedata_loaded = time.time()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build()
    time_kernel_compilation = time.time()

    print('execute kernel programs')
    evt = prg.find_nonce(queue, (10240, ), (1, ), cl_input_data.data, cl_output_data.data, cl_nonce.data, numpy.int32(len(data_str)), numpy.int32(3), numpy.int32(1024))
    print('wait for kernel executions')
    evt.wait()
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    time_before_readback = time.time()
    nonce = cl_nonce.get()
    time_after_readback = time.time()

    # hash_str = ''
    # for i in range(32):
    #     hash_str += '{:02x}'.format(output_hash[i])
    print('=' * 20)
    # print('hash: {}'.format(hash_str))
    print('nonce: {}'.format(nonce))
    print('=' * 20)

    print('Prepare host data took       : {}'.format(time_hostdata_loaded - start_time))
    print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation - time_hostdata_loaded))
    print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
    print('Compile kernel took          : {}'.format(time_kernel_compilation - time_devicedata_loaded))
    print('OpenCL elapsed time          : {}'.format(elapsed))
