import tensorflow as tf
import numpy as np
import random, urllib2, json
from threading import Thread
from flask import Flask
from flask_ask import Ask, session, request, statement, question, convert_errors

from voicelabs import VoiceInsights

app = Flask(__name__)
ask = Ask(app, '/')

BOARD_HEIGHT = 6
BOARD_WIDTH = 6
BOARD_SIZE = BOARD_HEIGHT * BOARD_WIDTH
PLANE_SIZE = 8
HIDDEN_UNITS = BOARD_SIZE
OUTPUT_UNITS = BOARD_SIZE
# voicelabs analytics
VI_APP_TOKEN = '464e7190-cc1e-11a6-0d08-02ddc10e4a8b'
vi = VoiceInsights()

# dashbot analytics
DASHBOT_API_KEY = 'QH8BnIP8WKu0sOCD3a89U3n9R08dAvfcTqyRSDD2'
DASHBOT_URL = 'https://tracker.dashbot.io/track?platform=generic&v=0.8.2-rest&type={}&apiKey={}'.format('outgoing', DASHBOT_API_KEY)

input_positions = tf.placeholder(tf.float32, shape=(1, BOARD_SIZE))
labels = tf.placeholder(tf.int64)
learning_rate = tf.placeholder(tf.float32, shape=[])

# 1st hidden layer
W1 = tf.Variable(tf.truncated_normal([BOARD_SIZE, HIDDEN_UNITS], stddev=0.1 / np.sqrt(float(BOARD_SIZE))))
b1 = tf.Variable(tf.zeros([1, HIDDEN_UNITS]))
h1 = tf.nn.relu(tf.matmul(input_positions, W1) + b1)

# 2nd hidden layer
W2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS, OUTPUT_UNITS], stddev=0.1 / np.sqrt(float(HIDDEN_UNITS))))
b2 = tf.Variable(tf.zeros([1, OUTPUT_UNITS]))
logits = tf.matmul(h1, W2) + b2
probabilities = tf.nn.softmax(logits)

init = tf.global_variables_initializer()
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, labels, name='xentropy')
train_step = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy)
# start() TF session
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Had to use Threading because dashbot easily causes Alexa to time out
def ping_dashbot(tts, userId):
    data = {'text': tts, 'userId': userId}
    dashbot_request = urllib2.Request(DASHBOT_URL)
    dashbot_request.add_header('Content-Type', 'application/json')
    payload = json.dumps(data)
    response = urllib2.urlopen(dashbot_request, payload)
    print 'INFO: response from dashbot: ' + str(response.code) + ' ' + response.msg

def init_game():
    # Initiate 2 boards to track game state:
    #
    # Cell states in agent board:
    #   2: The cell is covered by the plane and the user has successfully
    #      hit it.
    #   1: The cell is covered by the plane but the user hasn't tried to
    #      strike at the cell
    #   0: Unexplored -- the user hasn't tried to strike at the cell
    #  -1: The user strikes at the cell but it's a miss
    #
    # Cell states in user board:
    #   1: The agent strikes at the cell and the user confirms that it's
    #      a hit
    #   0: Unexplored -- the agent hasn't tried to strike at the cell
    #  -1: The agent strikes at the cell but it's a miss

    agent_board = []
    user_board = []
    for i in range(BOARD_HEIGHT):
        agent_board.append([0 for j in range(BOARD_WIDTH)])
        user_board.append([0 for j in range(BOARD_WIDTH)])


    # Populate the plane's position
    # First figure out the plane's orientation
    #   0: heading right
    #   1: heading up
    #   2: heading left
    #   3: heading down
    plane_orientation = random.randint(0, 3)

    # Figure out plane core's position as the '*' below
    #        |         | |
    #       -*-        |-*-
    #        |         | |
    #       ---
    if plane_orientation == 0:
        plane_core_row = random.randint(1, BOARD_HEIGHT - 2)
        plane_core_column = random.randint(2, BOARD_WIDTH - 2)
        # Populate the tail
        agent_board[plane_core_row][plane_core_column - 2] = 1
        agent_board[plane_core_row - 1][plane_core_column - 2] = 1
        agent_board[plane_core_row + 1][plane_core_column - 2] = 1
    elif plane_orientation == 1:
        plane_core_row = random.randint(1, BOARD_HEIGHT - 3)
        plane_core_column = random.randint(1, BOARD_WIDTH - 3)
        # Populate the tail
        agent_board[plane_core_row + 2][plane_core_column] = 1
        agent_board[plane_core_row + 2][plane_core_column + 1] = 1
        agent_board[plane_core_row + 2][plane_core_column - 1] = 1
    elif plane_orientation == 2:
        plane_core_row = random.randint(1, BOARD_HEIGHT - 2)
        plane_core_column = random.randint(1, BOARD_WIDTH - 3)
        # Populate the tail
        agent_board[plane_core_row][plane_core_column + 2] = 1
        agent_board[plane_core_row - 1][plane_core_column + 2] = 1
        agent_board[plane_core_row + 1][plane_core_column + 2] = 1
    else:
        plane_core_row = random.randint(2, BOARD_HEIGHT - 2)
        plane_core_column = random.randint(1, BOARD_WIDTH - 2)
        # Populate the tail
        agent_board[plane_core_row - 2][plane_core_column] = 1
        agent_board[plane_core_row - 2][plane_core_column + 1] = 1
        agent_board[plane_core_row - 2][plane_core_column - 1] = 1

    # Populate the cross
    agent_board[plane_core_row][plane_core_column] = 1
    agent_board[plane_core_row + 1][plane_core_column] = 1
    agent_board[plane_core_row - 1][plane_core_column] = 1
    agent_board[plane_core_row][plane_core_column + 1] = 1
    agent_board[plane_core_row][plane_core_column - 1] = 1

    return agent_board, user_board, 0, 0

def predict_next_strike_position(game_board):
    flattened_game_board = sum(game_board, []) # flattened board
    tmp = sess.run(probabilities, feed_dict={input_positions: [flattened_game_board]})
    probs = tmp[0]
    probs = [p * (game_board[index/BOARD_WIDTH][index%BOARD_WIDTH] == 0) for index, p in enumerate(probs)]
    probs = [p / sum(probs) for p in probs]
    strike_pos = np.argmax(probs)
    x = strike_pos / BOARD_WIDTH
    y = strike_pos % BOARD_WIDTH
    return x, y


@ask.on_session_started
def new_session():
    print '*****New session started*****'
    vi.initialize(VI_APP_TOKEN, session)


@ask.launch
def start_plane_strike():
    print '*****Plane Strike launched*****'
    tts = 'Welcome to Plane Strike. If this is the '                       \
              'first time you play this game, please say \'Help\', so '    \
              'that I can give you some more information to get start()ed. ' \
              'If you already know how this game works, please let '       \
              'me know your first strike position. '                       \
              'For example, row 2, column 3.'
    agent_board, user_board, total_hits_by_agent, total_hits_by_user = init_game()
    session.attributes['agent_board']         = agent_board
    session.attributes['user_board']          = user_board
    session.attributes['total_hits_by_agent'] = total_hits_by_agent
    session.attributes['total_hits_by_user']  = total_hits_by_user
    session.attributes['agent_strike_row']    = None
    session.attributes['agent_strike_column'] = None
    session.attributes['expected_intents']    = ['USER_PROVIDE_STRIKE_POSITION']

    vi.track('start()_PLANE_STRIKE', request, tts)
    Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
    return question(tts)

# agent hit
@ask.intent('USER_CONFIRM_HIT')
def user_confirm_hit():
    if 'USER_CONFIRM_HIT' not in session.attributes['expected_intents']:
        print '*****Wrong ASR*****'
        print 'Should be ' + ''.join(session.attributes['expected_intents']) + \
              ' handlers but is actually in USER_CONFIRM_HIT handler'
        tts = 'I\'m sorry. Please give 2 numbers between 1 and 6 as your strike position. For example, row 2 column 3.'
        session.attributes['expected_intents'] = ['USER_PROVIDE_STRIKE_POSITION']

        vi.track('USER_CONFIRM_HIT', request, tts)
        Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
        return question(tts)

    print '*****User confirms hit*****'
    user_board          = session.attributes['user_board']
    agent_strike_row    = session.attributes['agent_strike_row']
    agent_strike_column = session.attributes['agent_strike_column']

    session.attributes['total_hits_by_agent'] = session.attributes['total_hits_by_agent'] + 1
    user_board[agent_strike_row][agent_strike_column] = 1

    if session.attributes['total_hits_by_agent'] == PLANE_SIZE:
        tts = 'Awesome! I believe I have won the '                    \
                  'game by hitting all ' + str(PLANE_SIZE) + ' '      \
                  'parts of your plane. Thank you '                   \
                  'for playing!'
        session.attributes['expected_intents'] = []

        vi.track('USER_CONFIRM_HIT', request, tts)
        Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
        return statement(tts)
    else:
        tts = 'Great! What\'s your next strike '                      \
                  'position? Tell me the row '                        \
                  'and column number. For example, '                  \
                  'row 3, column 4'
        session.attributes['expected_intents'] = ['USER_PROVIDE_STRIKE_POSITION']

        vi.track('USER_CONFIRM_HIT', request, tts)
        Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
        return question(tts)

# agent missed
@ask.intent('USER_CONFIRM_MISS')
def user_confirm_miss():
    if 'USER_CONFIRM_MISS' not in session.attributes['expected_intents']:
        print '*****Wrong ASR*****'
        print 'Should be ' + ''.join(session.attributes['expected_intents']) + \
              ' handlers but is actually in USER_CONFIRM_MISS handler'
        tts = 'I\'m sorry. Please give 2 numbers between 1 and 6 as your strike position. For example, row 2 column 3.'
        session.attributes['expected_intents'] = ['USER_PROVIDE_STRIKE_POSITION']

        vi.track('USER_CONFIRM_MISS', request, tts)
        Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
        return question(tts)

    print '*****User confirms miss*****'
    user_board          = session.attributes['user_board']
    agent_strike_row    = session.attributes['agent_strike_row']
    agent_strike_column = session.attributes['agent_strike_column']

    user_board[agent_strike_row][agent_strike_column] = -1
    tts = 'Too bad I missed! What\'s your next '                     \
              'strike position? Tell me the row '                    \
              'and column number. For example, 5, 1'
    session.attributes['expected_intents'] = ['USER_PROVIDE_STRIKE_POSITION']

    vi.track('USER_CONFIRM_MISS', request, tts)
    Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
    return question(tts)


@ask.intent('USER_PROVIDE_STRIKE_POSITION', convert={'x': int, 'y': int})
def user_provide_strike_position(x, y):
    if 'USER_PROVIDE_STRIKE_POSITION' not in session.attributes['expected_intents']:
        print '*****Wrong ASR*****'
        print 'Should be ' + ''.join(session.attributes['expected_intents']) + \
              ' handlers but is actually in USER_PROVIDE_STRIKE_POSITION handler'
        tts = 'I\'m sorry. Could you confirm if I hit or missed?'
        session.attributes['expected_intents'] = ['USER_CONFIRM_MISS', 'USER_CONFIRM_HIT']

        vi.track('USER_PROVIDE_STRIKE_POSITION', request, tts)
        Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
        return question(tts)

    print '*****User provides strike position****'
    agent_board = session.attributes['agent_board']
    user_board = session.attributes['user_board']

    if x in convert_errors or y in convert_errors:
        tts = 'Could you say that again?'
        session.attributes['expected_intents'] = ['USER_PROVIDE_STRIKE_POSITION']

        vi.track('USER_PROVIDE_STRIKE_POSITION', request, tts)
        Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
        return question(tts)

    if (not x) or (not y) or                                        \
       x > BOARD_HEIGHT or x < 1 or                                 \
       y > BOARD_WIDTH  or y < 1:
        tts = 'I\'m sorry. Please give 2 numbers between 1 and 6 as your strike position. For example, row 2 column 3.'
        session.attributes['expected_intents'] = ['USER_PROVIDE_STRIKE_POSITION']

        vi.track('USER_PROVIDE_STRIKE_POSITION', request, tts)
        Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
        return question(tts)
    else:
        # shift by 1
        user_strike_row = x - 1
        user_strike_column = y - 1
        if agent_board[user_strike_row][user_strike_column] == 1:
            session.attributes['total_hits_by_user'] = session.attributes['total_hits_by_user'] + 1
            agent_board[user_strike_row][user_strike_column] = 2
            print '*** total user hits: ' + str(session.attributes['total_hits_by_user'])
            if session.attributes['total_hits_by_user'] == PLANE_SIZE:
                tts = 'Congratulations! You have won the game!'
                session.attributes['expected_intents'] = []

                vi.track('USER_PROVIDE_STRIKE_POSITION', request, tts)
                Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
                return statement(tts)
            else:
                tts = 'Oh, row ' + str(user_strike_row+1) + ''           \
                          ' column ' + str(user_strike_column+1) + ''    \
                          ' was a hit! Now it\'s my turn. Strike at '
        elif agent_board[user_strike_row][user_strike_column] == 2:
            tts = 'You\'ve just made a repeat shot! Row ' +              \
                      str(user_strike_row + 1) + ' column ' +            \
                      str(user_strike_column + 1) + ' was a hit. '       \
                      'Now it\'s my turn. Strike at '
        elif agent_board[user_strike_row][user_strike_column] == -1:
            tts = 'You\'ve just made a repeat shot! Row ' +              \
                      str(user_strike_row+1) + ' column ' +              \
                      str(user_strike_column+1) + ' was a miss. '        \
                      'Now it\'s my turn. Strike at '
        else:
            tts = 'Unfortunately row ' + str(user_strike_row+1) + ' '    \
                      ' column ' + str(user_strike_column+1) + ''        \
                      ' was a miss! Now it\'s my turn. Strike at '
        agent_strike_row, agent_strike_column = predict_next_strike_position(user_board)
        tts = tts + 'row ' + str(agent_strike_row+1) + ' column ' + str(agent_strike_column+1) + '. Did I hit your plane?'
        session.attributes['agent_strike_row'] = agent_strike_row
        session.attributes['agent_strike_column'] = agent_strike_column
        session.attributes['expected_intents'] = ['USER_CONFIRM_MISS', 'USER_CONFIRM_HIT']

        vi.track('USER_PROVIDE_STRIKE_POSITION', request, tts)
        Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
        return question(tts)


# stop game
@ask.intent('AMAZON.CancelIntent')
def stop_game():
    print '*****User cancels*****'
    tts = 'Thank you for playing Plane Strike!'
    session.attributes['expected_intents'] = []

    vi.track('AMAZON.CancelIntent', request, tts)
    Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
    return statement(tts)

# stop game
@ask.intent('AMAZON.StopIntent')
def stop_game():
    print '*****User stops****'
    tts = 'Thank you for playing Plane Strike!'
    session.attributes['expected_intents'] = []

    vi.track('AMAZON.StopIntent', request, tts)
    Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
    return statement(tts)

# help
@ask.intent('AMAZON.HelpIntent')
def help():
    print '*****User asks for help*****'
    tts = 'Thank you for playing this game! If you know how the Battleship game works, '      \
                         'you pretty much already know how this game works: you and the computer '      \
                         'will take turns to strike at a cell in each other\'s board; whoever hits '    \
                         'all eight cells first wins. Please visit plane strike dot wordpress dot com'  \
                         'for more detailed information. Once you are ready to take the challenge '     \
                         'powered by machine learning, let me know your strike position by speaking '   \
                         'out your row and column number.'
    session.attributes['expected_intents'] = ['USER_PROVIDE_STRIKE_POSITION']

    # TODO: context specific help

    vi.track('AMAZON.HelpIntent', request, tts)
    Thread(target=ping_dashbot, args=(tts, session.user.userId)).start()
    return question(tts)

@ask.session_ended
def session_ended():
    print '*****Session ending*****'
    return "", 200

if __name__ == '__main__':
    app.run(debug=True)