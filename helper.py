import matplotlib.pyplot as plt
from IPython import display
import pickle
import os.path

plt.ion()
FILES_LOCATION = 'neat-checkpoint/'
# dump information to that file
def create_results_file():
    file = open(FILES_LOCATION+'gen-neat-results', 'wb')
    pickle.dump(([],[]), file)
    file.close()

def plot(max_fit, mean_fit):
    if not os.path.exists(FILES_LOCATION+'gen-neat-results'):
        create_results_file()
        print('no exists')
    # open a file, where you stored the pickled data
    file = open(FILES_LOCATION+'gen-neat-results', 'rb')
    (max_scores, mean_scores) = pickle.load(file)
    file.close()

    max_scores.append(max_fit)
    mean_scores.append(mean_fit)


    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness')
    #plt.bar(range(len(max_scores)), max_scores, label='BEST Fitness', color='#98c8fa')
    plt.plot(mean_scores, label='MEAN Fitness', linewidth=4, color='orange')
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(max_scores)-1, max_scores[-1], str(round(max_scores[-1],2)))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1],2)))
    
    file = open(FILES_LOCATION+'gen-neat-results', 'wb')
    pickle.dump((max_scores, mean_scores), file)
    file.close()

    plt.show(block=False)
    plt.savefig(FILES_LOCATION+'training_plot.png')
    plt.pause(.1)
