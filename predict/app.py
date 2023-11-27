from utils import choose_best_move, model
import chess
from utils import evaluate_position
import chess.svg
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # Assuming you have an HTML file for input
        return render_template('index.html')

    if request.method == 'POST':
        # Get the FEN string from the form
        fen_string = request.form['fen_input']
        score = evaluate_position(model, chess.Board(fen_string))

        # Generate SVG image of the board
        board = chess.Board(fen_string)
        board_svg = chess.svg.board(board=board, size=350)

        best_move = choose_best_move(board, depth=2)

        return render_template('result.html', fen=fen_string, score=score, board_svg=board_svg, best_move=best_move)


if __name__ == '__main__':
    app.run(debug=True)

# test_example = 'r1b1kbnr/pppN1ppp/4p3/3p2q1/8/2N1P3/PPn2PPP/R1BQKB1R w KQkq - 0 1'
# value = evaluate_position(model, test_example)
# print(value)
