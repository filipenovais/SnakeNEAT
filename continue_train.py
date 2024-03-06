import os
import neat
from snake import SnakeGameAI
import random
import time
import numpy as np
from helper import plot, FILES_LOCATION
import threading
from queue import Queue


def run_winner(genome, config):
    game = SnakeGameAI(showBool=True)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    genome.fitness = 0
    game.reset()
    sim_time = 5000
    last_timee = 0
    last_score = 0
    for timee in range(sim_time):
        game_state = game.get_game_state()
        #print(game_state)

        final_move = [0,0,0]
        move = random.randint(0, 2)
        output = net.activate(game_state)
        final_move[np.argmax(softmax(output))] = 1
        reward, done, score = game.play_step(final_move)
        if score > last_score:
            last_timee = 0
            last_score = score
        if done:
            genome.fitness = score*10 + 100*((timee-1)/sim_time)
            break
        elif last_timee == 250:
            genome.fitness = score
            print('*** LOOP FOUND ***')
            break
        elif timee == sim_time-1:
            genome.fitness =  5000
            print('*** TIME ENDED ***')
            break
        last_timee += 1
        #print(last_timee)
    print('WINNER FITNESS: ' + str(genome.fitness))

def eval_nn(genome, config):
    game = SnakeGameAI(showBool=True)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    genome.fitness = 0
    game.reset()
    sim_time = 50000
    last_timee = 0
    last_score = 0
    for timee in range(sim_time):
        game_state = game.get_game_state()
        #print(game_state)

        final_move = [0,0,0]
        move = random.randint(0, 2)
        output = net.activate(game_state)
        final_move[np.argmax(softmax(output))] = 1
        reward, done, score = game.play_step(final_move)
        if score > last_score:
            last_timee = 0
            last_score = score
        if done:
            if score > 10:
                genome.fitness = 100+(score-10)*5 - last_timee/500
            else:
                genome.fitness = score*10
            break
        elif last_timee == 500:
            genome.fitness = score*10
            print('*** LOOP FOUND ***')
            break
        elif timee == sim_time-1:
            genome.fitness =  5000
            print('*** TIME ENDED ***')
            break
        last_timee += 1
        #print(last_timee)
    print('FITNESS: ' + str(genome.fitness))
    all_fitness.put(genome.fitness)
    return genome


def func_thread(x, results):
    rand_int_var = random.randint(1, 5)
    print(rand_int_var)
    time.sleep(rand_int_var)
    print('end', rand_int_var)
    q.put(rand_int_var)

def eval_genomes(genomes, config):
    plot_scores = []
    plot_mean_fitness = []

    '''thread_list=[]
    for genome_id, genome in genomes:
        thread = threading.Thread(target=eval_nn, args=(genome, config))
        thread_list.append(thread)
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()

    print('end')
    print(list(all_fitness.queue))
    exit()'''
    for genome_id, genome in genomes:
        genome = eval_nn(genome, config)

    #get_winner()
    #run_winner(genome, config)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint/neat-checkpoint-'+'3419')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    # Run for 200 generation.
    p.run(eval_genomes, 2000)    






if __name__ == '__main__':
    all_fitness = Queue()
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, FILES_LOCATION+'config-feedforward')
    run(config_path)
