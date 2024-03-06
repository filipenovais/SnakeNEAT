import os
import neat
from snake import SnakeGameAI
import random
from helper import plot, FILES_LOCATION, create_results_file
import time
import numpy as np


def eval_nn(genome, config):
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

        final_move = [0, 0, 0]
        move = random.randint(0, 2)
        output = net.activate(game_state)
        final_move[np.argmax(softmax(output))] = 1
        reward, done, score = game.play_step(final_move)
        if score > last_score:
            last_timee = 0
            last_score = score
        if done:
            #genome.fitness = score*10 + 100*((timee-1)/sim_time)
            genome.fitness = score + timee*0.01
            break
        elif last_timee == 350:
            genome.fitness = score*0.1
            print('*** LOOP FOUND ***')
            break
        elif timee == sim_time-1:
            genome.fitness = 500
            print('*** TIME ENDED ***')
            break
        last_timee += 1
        #print(last_timee)
    return genome

def eval_genomes(genomes, config):
    plot_scores = []
    plot_mean_fitness = []
    all_fitness = []
    genome_count = 0
    for genome_id, genome in genomes:
        genome = eval_nn(genome, config)
        all_fitness.append(genome.fitness)
        
        strgenome = "("+str(genome_count+1)+"/"+str(len(genomes))+")"
        print(strgenome+' FITNESS: ' + str(genome.fitness))
        genome_count += 1 

    #plot(max(all_fitness), sum(all_fitness)/len(all_fitness))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 2000 generations.
    gen_numb = 5000
    p.add_reporter(neat.Checkpointer(10))
    winner = p.run(eval_genomes, gen_numb)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-'+str(gen_numb-1))
    #p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    create_results_file()
    config_path = os.path.join(local_dir, FILES_LOCATION+'config-feedforward')
    run(config_path)
