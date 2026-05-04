from utils.model import train_model, show_model_score
from utils.cube import visualize_scramble, execute_move, is_cube_solved, visualize_scramble_terminal

MAX_NUM_MOVES = 1000

def main():
    model, X_test, y_test = train_model("cfop-dataset-processed/dataset.pkl", 100000)
    show_model_score(model, X_test, y_test)

    state = X_test[:1]

    number_of_moves = 0
    while not is_cube_solved(state):
        if number_of_moves > MAX_NUM_MOVES:
            break

        prediction = model.predict(state)[0]
        state = execute_move(prediction, state)
        print(f"Prediction: {prediction}")
        number_of_moves += 1

    # visualize_scramble(random_state)
    # print(state)
    print(f"Took {number_of_moves} moves")
    visualize_scramble_terminal(state)


if __name__ == '__main__':
    main()
