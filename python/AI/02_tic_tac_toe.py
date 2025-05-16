# write code for tic tac toe game
import random
import os

def print_board(board):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("--+---+--")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("--+---+--")
    print(f"{board[6]} | {board[7]} | {board[8]}")

def check_winner(board, player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]             # diagonals
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

def is_draw(board):
    return all(cell in ['X', 'O'] for cell in board)

def player_move(board):
    while True:
        try:
            move = int(input("Enter your move (1-9): ")) - 1
            if board[move] not in ['X', 'O']:
                board[move] = 'X'
                break
            else:
                print("Cell already taken. Try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number between 1 and 9.")

def computer_move(board):
    available_moves = [i for i in range(9) if board[i] not in ['X', 'O']]
    move = random.choice(available_moves)
    board[move] = 'O'

def main():
    board = [str(i + 1) for i in range(9)]
    print_board(board)

    while True:
        player_move(board)
        print_board(board)
        if check_winner(board, 'X'):
            print("Congratulations! You win!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

        print("Computer's turn...")
        computer_move(board)
        print_board(board)
        if check_winner(board, 'O'):
            print("Computer wins! Better luck next time.")
            break
        if is_draw(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    main()