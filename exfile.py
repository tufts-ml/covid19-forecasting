print('hello world')
# import turicreate as tc 

# print(tc)

print('asdasdfeeaasdfsdfsasdfasf')

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', default='MA')
    parser.add_argument('--start_date', default='20201111', type=str)
    parser.add_argument('--end_training_date', default='20210111', type=str)
    parser.add_argument('--end_date', default='20210211', type=str)
    args = parser.parse_args()

    print(args.state, args.start_date, args.end_training_date)