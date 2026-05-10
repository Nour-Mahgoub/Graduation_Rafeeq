import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/nour-mahgoub/Graduation_Rafeeq/install/serial_node'
