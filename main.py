from CubeML.utils.model import train_model, show_model_score
from CubeML.utils.cube import visualize_scramble, execute_move, is_cube_solved, visualize_scramble_terminal

MAX_NUM_MOVES = 1000

def main():
    model, X_test, y_test = train_model("autoencoder", "cfop-dataset-processed/dataset_no_prime.pkl", 10000)
    show_model_score(model, X_test, y_test)

    state = X_test[:1]
    visualize_scramble_terminal(state)

    number_of_moves = 0
    last_4_moves = []
    while not is_cube_solved(state):
        if number_of_moves > MAX_NUM_MOVES:
            break

        prediction = model.predict(state)[0]
        confidence = model.predict_proba(state).max()
        print(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
        # print(f"Prediction: {prediction}")

        last_4_moves.append(prediction)
        if len(last_4_moves) > 4:
            last_4_moves.pop(0)
        if len(set(last_4_moves)) == 1 and len(last_4_moves) >= 4:
            print("System converged")
            break

        state = execute_move(str(prediction), state)
        number_of_moves += 1

        input()
        visualize_scramble_terminal(state)

    # visualize_scramble(random_state)
    # print(state)
    print(f"Took {number_of_moves} moves")
    visualize_scramble_terminal(state)


if __name__ == '__main__':
    main()
