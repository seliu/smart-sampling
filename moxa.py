import numpy
import scipy.signal
import thinkdsp

def filter_data_batch( batch_data, filter_type = 'gaussian', window_size = 11, std = 1 ):
    filtered_data = [ filter_data( data, filter_type = filter_type, window_size = window_size, std = std ) for data in batch_data ]
    filtered_data = numpy.array( filtered_data )
    return filtered_data

def filter_data( data, filter_type = 'gaussian', window_size = 11, std = 1 ):
    gaussian = scipy.signal.gaussian( M = window_size, std = std )
    gaussian /= sum( gaussian )
    convolved = numpy.convolve( data, gaussian, mode = 'same' )
    return convolved

def get_test_data( freq, n_offset = 10, random_seed = 42, shape_list = [ 'sawtooth', 'square', 'triangle', 'cosine', 'chirp' ], noise_ratio = 0 ):
    numpy.random.seed( random_seed )
    offset_list = 2 * numpy.pi * numpy.random.uniform( size = n_offset )
    data_x, data_y = get_wave_data( freq, offset_list, duration = 0.1, framerate = 40000, shape_list = shape_list )
    data_x = numpy.array( data_x )
    if noise_ratio != 0:
        data_x += numpy.random.standard_normal( data_x.shape ) * noise_ratio
        # data_x /= 1 + noise_ratio
    data_y = numpy.expand_dims( data_y, axis = 1 )
    return data_x, data_y

def get_train_data_v0( freq_lower, freq_upper, freq_step, n_offset = 10, shape_list = [ 'sawtooth', 'square', 'triangle' ] ):
    data_x, data_y = [], []
    offset_list = 2 * numpy.pi * ( numpy.arange( n_offset ) / n_offset )
    for freq in range( freq_lower, freq_upper + 1, freq_step ):
        data_x_one_freq, data_y_one_freq = get_wave_data( freq, offset_list, duration = 0.1, framerate = 40000, shape_list = shape_list )
        data_x.extend( data_x_one_freq )
        data_y.extend( data_y_one_freq )
    data_x = numpy.array( data_x )
    data_y = numpy.expand_dims( data_y, axis = 1 )
    return data_x, data_y

def get_train_data_v2( n_offset = 10, shape_list = [ 'sawtooth', 'square', 'triangle', 'cosine' ] ):
    data_x, data_y = [], []
    offset_list = 2 * numpy.pi * ( numpy.arange( n_offset ) / n_offset )
    for freq in range( 10, 2000, 1 ):
        data_x_one_freq, data_y_one_freq = get_wave_data( freq, offset_list, duration = 0.1, framerate = 40000, shape_list = shape_list )
        data_x.extend( data_x_one_freq )
        data_y.extend( data_y_one_freq )
    for freq in range( 2000, 4001, 2 ):
        data_x_one_freq, data_y_one_freq = get_wave_data( freq, offset_list, duration = 0.1, framerate = 40000, shape_list = shape_list )
        data_x.extend( data_x_one_freq )
        data_y.extend( data_y_one_freq )
    data_x = numpy.array( data_x )
    data_y = numpy.expand_dims( data_y, axis = 1 )
    return data_x, data_y

def get_train_data_v3( n_offset = 10, shape_list = [ 'sawtooth', 'square', 'triangle', 'cosine' ] ):
    data_x, data_y = [], []
    offset_list = 2 * numpy.pi * ( numpy.arange( n_offset ) / n_offset )
    for freq in range( 10, 1000, 1 ):
        data_x_one_freq, data_y_one_freq = get_wave_data( freq, offset_list, duration = 0.1, framerate = 40000, shape_list = shape_list )
        data_x.extend( data_x_one_freq )
        data_y.extend( data_y_one_freq )
    for start in range( 1000, 6000, 500 ):
        step = start // 500 # that is, take 0.2% as the step
        end = start + 500
        for freq in range( start, end, step ):
            data_x_one_freq, data_y_one_freq = get_wave_data( freq, offset_list, duration = 0.1, framerate = 40000, shape_list = shape_list )
            data_x.extend( data_x_one_freq )
            data_y.extend( data_y_one_freq )
    data_x = numpy.array( data_x )
    data_y = numpy.expand_dims( data_y, axis = 1 )
    return data_x, data_y

def get_wave_data( freq, offset_list, duration = 0.1, framerate = 40000, shape_list = [ 'sawtooth', 'square', 'triangle', 'cosine' ] ):
    data_x, data_y = [], []

    for offset in offset_list:
        duration = 0.1
        framerate = 40000

        # Sawtooth
        if 'sawtooth' in shape_list:
            data = get_sawtooth_wave_data( freq = freq, offset = offset, duration = duration, framerate = framerate, noise_ratio = 0 )
            data_x.append( data )
            data_y.append( freq )

        # Square
        if 'square' in shape_list:
            data = get_square_wave_data( freq = freq, offset = offset, duration = duration, framerate = framerate, noise_ratio = 0 )
            data_x.append( data )
            data_y.append( freq )

        # Triangle
        if 'triangle' in shape_list:
            data = get_triangle_wave( freq = freq, offset = offset, duration = duration, framerate = framerate )
            data_x.append( data )
            data_y.append( freq )

        # Cosine
        if 'cosine' in shape_list:
            data = get_cosine_wave( freq = freq, offset = offset, duration = duration, framerate = framerate )
            data_x.append( data )
            data_y.append( freq )

        # Mixed: Sawtooth ( offset 0 ) + Square ( offset 2*Pi*0.2 )
        if 'sawtooth+square' in shape_list:
            data = get_sawtooth_square_wave( freq = freq, offset = offset, duration = duration, framerate = framerate )
            data_x.append( data )
            data_y.append( freq )

        if 'chirp' in shape_list:
            data = get_chirp_wave_data( freq = freq, offset = offset, duration = duration, framerate = framerate )
            data_x.append( data )
            data_y.append( freq )

    return data_x, data_y

def get_sawtooth_wave_data( freq, offset, duration = 0.1, framerate = 40000, noise_ratio = 0 ):
    signal = thinkdsp.SawtoothSignal( freq = freq, amp = 1.0, offset = offset )
    wave = signal.make_wave( duration = duration, start = 0, framerate = framerate )
    if noise_ratio != 0:
        wave.ys += numpy.random.standard_normal( wave.ys.shape ) * noise_ratio
    return wave.ys

def get_square_wave_data( freq, offset, duration = 0.1, framerate = 40000, noise_ratio = 0 ):
    signal = thinkdsp.SquareSignal( freq = freq, amp = 1.0, offset = offset )
    wave = signal.make_wave( duration = duration, start = 0, framerate = framerate )
    if noise_ratio != 0:
        wave.ys += numpy.random.standard_normal( wave.ys.shape ) * noise_ratio
    return wave.ys
def get_square_wave( freq, offset, duration = 0.1, framerate = 40000 ):
    signal = thinkdsp.SquareSignal( freq = freq, amp = 1.0, offset = offset )
    wave = signal.make_wave( duration = duration, start = 0, framerate = framerate )
    return wave.ys

def get_triangle_wave( freq, offset, duration = 0.1, framerate = 40000 ):
    signal = thinkdsp.TriangleSignal( freq = freq, amp = 1.0, offset = offset )
    wave = signal.make_wave( duration = duration, start = 0, framerate = framerate )
    return wave.ys

def get_cosine_wave( freq, offset, duration = 0.1, framerate = 40000 ):
    signal = thinkdsp.CosSignal( freq = freq, amp = 1.0, offset = offset )
    wave = signal.make_wave( duration = duration, start = 0, framerate = framerate )
    return wave.ys

def get_sawtooth_square_wave( freq, offset, duration = 0.1, framerate = 40000 ):
    signal_sawtooth = thinkdsp.SawtoothSignal( freq = freq, amp = 1.0, offset = offset )
    signal_square = thinkdsp.SquareSignal( freq = freq, amp = 1.0, offset = offset + 2 * numpy.pi * 0.2 )
    components = [ signal_sawtooth, signal_square ]
    signal = thinkdsp.SumSignal( *components )
    wave = signal.make_wave( duration = duration, start = 0, framerate = 40000 )
    wave.ys = wave.ys / len( components ) # normalization
    return wave.ys

def get_chirp_wave_data( freq, offset, duration = 0.1, framerate = 40000, noise_ratio = 0 ):
    signal = thinkdsp.Chirp( start = freq * 0.5, end = 1.5, amp = 1.0 )
    wave = signal.make_wave( duration = duration, start = 0, framerate = framerate )
    if noise_ratio != 0:
        wave.ys += numpy.random.standard_normal( wave.ys.shape ) * noise_ratio
    return wave.ys
