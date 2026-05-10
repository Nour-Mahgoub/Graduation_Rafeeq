import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/nour-mahgoub/testingGrad/install/view_robot_pkg'
