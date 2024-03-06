import os
import neat
from snake import SnakeGameAI
import random
import time
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



def eval_dummy_genome_nn(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ignored_output = net.activate(list(range(11)))
    return 0.0


def eval_dummy_genomes_nn(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_dummy_genome_nn(genome, config)


def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint/neat-checkpoint-'+'415')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    # Run for 1 generation.
    p.run(eval_dummy_genomes_nn, 1)


    winner = stats.best_genome()
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Display the winning genome.
    game = SnakeGameAI(showBool=True)
    while True:
        game.reset()
        t = 10000
        for timee in range(t):
            game_state = game.get_game_state()

            move = random.randint(0, 2)
            
            output = winner_net.activate(game_state)
            final_move = [0,0,0]
            final_move[np.argmax(softmax(output))] = 1
            reward, done, score = game.play_step(final_move)
            if done:
                break
            elif timee == (t-1):
                print(' chromossome died!!  -------- I wish I had more time! :( ')
                break





if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-checkpoint/config-feedforward')
    run(config_path)
