import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import midi

graph1 = tf.Graph() #make sure all variables are initialized to the same graph scope
with graph1.as_default():
    lowerBound = 24
    upperBound = 102
    span = upperBound - lowerBound

    ### HyperParameters

    lowest_note = lowerBound
    highest_note = upperBound
    note_range = highest_note - lowest_note

    num_timesteps = 5
    n_visible = 2 * note_range * num_timesteps
    n_hidden = 50

    num_epochs = 400  # For each epoch we go through the entire data set.
    batch_size = 200  # Number of training examples that we are going to send through the RBM at a time.
    lr = tf.constant(0.005, tf.float32)  # The learning rate of our model

    ### Variables:
    # Next, let's look at the variables we're going to use:

    x = tf.placeholder(tf.float32, [None, n_visible], name="x")  # Data placeholder variable
    W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01),
                    name="W")  # Weight matrix stores edge weights
    bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="bh"))  # Hidden layer bias vector
    bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv"))  # Visible layer bias vector


    # Midi helper functions
    def midiToNoteStateMatrix(midifile, squash=True, span=span):
        print(midifile)
        pattern = midi.read_midifile(midifile)

        timeleft = [track[0].tick for track in pattern]

        posns = [0 for track in pattern]

        statematrix = []
        time = 0

        state = [[0, 0] for x in range(span)]
        statematrix.append(state)
        condition = True
        while condition:
            if time % (pattern.resolution / 4) == (pattern.resolution / 8):
                # Crossed a note boundary. Create a new state, defaulting to holding notes
                oldstate = state
                state = [[oldstate[x][0], 0] for x in range(span)]
                statematrix.append(state)
            for i in range(len(timeleft)):  # For each track
                if not condition:
                    break
                while timeleft[i] == 0:
                    track = pattern[i]
                    pos = posns[i]

                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                            pass
                            # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                        else:
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch - lowerBound] = [0, 0]
                            else:
                                state[evt.pitch - lowerBound] = [1, 1]
                    elif isinstance(evt, midi.TimeSignatureEvent):
                        if evt.numerator not in (2, 4):
                            # We don't want to worry about non-4 time signatures. Bail early!
                            # print "Found time signature event {}. Bailing!".format(evt)
                            out = statematrix
                            condition = False
                            break
                    try:
                        timeleft[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        timeleft[i] = None

                if timeleft[i] is not None:
                    timeleft[i] -= 1

            if all(t is None for t in timeleft):
                break

            time += 1

        S = np.array(statematrix)
        statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
        statematrix = np.asarray(statematrix).tolist()
        return statematrix


    def newTrack():
        return midi.Track()


    def noteStateMatrixToTrack(statematrix, track, span=span):
        statematrix = np.array(statematrix)
        if not len(statematrix.shape) == 3:
            statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
        statematrix = np.asarray(statematrix)

        span = upperBound - lowerBound
        tickscale = 55

        lastcmdtime = 0
        prevstate = [[0, 0] for x in range(span)]
        for time, state in enumerate(statematrix + [prevstate[:]]):
            offNotes = []
            onNotes = []
            for i in range(span):
                n = state[i]
                p = prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] == 1:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] == 1:
                    onNotes.append(i)
            for note in offNotes:
                track.append(midi.NoteOffEvent(tick=(time - lastcmdtime) * tickscale, pitch=note + lowerBound))
                lastcmdtime = time
            for note in onNotes:
                track.append(
                    midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale, velocity=40, pitch=note + lowerBound))
                lastcmdtime = time

            prevstate = state

        return track


    def trackToMidi(track, name="sample"):
        pattern = midi.Pattern()
        pattern.append(track)
        track.append(midi.EndOfTrackEvent(tick=1))
        midi.write_midifile(name, pattern)


    def get_songs(path1, path2):
        files = [path1, path2]
        songs = []
        for f in tqdm(files):
            try:
                song = np.array(midiToNoteStateMatrix(f))
                if np.array(song).shape[0] > 50:
                    songs.append(song)
            except Exception as e:
                raise e
        return songs


    #### Helper functions.

    def sample(probs):
        # Returns a random vector of 0s and 1s sampled from the input vector of probabilities
        return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


    # This function runs the gibbs chain. We will call this function in two places:
    # During the training update step
    # During sampling of our music segments from the trained RBM
    def gibbs_sample(k):
        # Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
        def gibbs_step(count, k, xk):
            # A single gibbs step. The visible values initialized to xk
            hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))  # Propagate the visible values to sample the hidden values
            xk = sample(tf.sigmoid(
                tf.matmul(hk, tf.transpose(W)) + bv))  # Propagate the hidden values to sample the visible values
            return count + 1, k, xk

        # Gibbs steps for k iterations
        ct = tf.constant(0)  # tensorflow counter
        [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                       gibbs_step, [ct, tf.constant(k), x])
        x_sample = tf.stop_gradient(x_sample)
        return x_sample


    ### Training Update Code
    # Now we implement the contrastive divergence algorithm. First, we get the samples of x and h from the probability distribution
    # The sample of x
    x_sample = gibbs_sample(1)
    # The sample of the hidden nodes, starting from the visible state of x
    h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
    # The sample of the hidden nodes, starting from the visible state of x_sample
    h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

    # Next, we update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
    size_bt = tf.cast(tf.shape(x)[0], tf.float32)
    W_adder = tf.multiply(lr / size_bt,
                          tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
    bv_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
    bh_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
    # When we do sess.run(updt), TensorFlow will run all 3 update steps
    updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]


    def generate_midi(path1, path2, songName):
        songs = get_songs(path1, path2)
        print("{} songs processed".format(len(songs)))


        with tf.Session(graph=graph1) as sess:
            # Train, and initialize the variables of, the model
            init = tf.global_variables_initializer()
            sess.run(init)
            # Run through all of the training data num_epochs times
            for epoch in tqdm(range(num_epochs)):
                for song in songs:
                    # Size of each song is num_timesteps x 2*note_range
                    song = np.array(song)
                    song = song[:int(np.floor(song.shape[0] / num_timesteps) * num_timesteps)]
                    song = np.reshape(song, [int(song.shape[0] / num_timesteps), int(song.shape[1] * num_timesteps)])
                    # Train on batch_size examples
                    for i in range(1, len(song), batch_size):
                        tr_x = song[i:i + batch_size]
                        sess.run(updt, feed_dict={x: tr_x})

            # Gibbs chain with visible nodes initialized to 0
            sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visible))})
            notRepeat = True
            track = newTrack()
            for i in range(sample.shape[0]):
                if not any(sample[i, :]):
                    continue
                # Reshape the vector to be time x notes, and save the vector as a midi file
                S = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
                track = noteStateMatrixToTrack(S, track)
            trackToMidi(track, songName)

