from build_tag_similiarity_network import build_network
from build_user_availibility_network import build_user_availibility_network
from build_user_basic_profile_network import build_user_basic_profile_network
from build_user_expertise_network import build_user_expertise_network
from build_user_similiarity_network import build_user_similiarity_network


if __name__ == '__main__':
    # build_network()
    # build_user_similiarity_network()

    print('building profile network...')
    build_user_basic_profile_network()

    print('building availability network...')
    build_user_availibility_network()

    print('building expertise network...')
    build_user_expertise_network()
