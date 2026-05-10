from setuptools import find_packages, setup

package_name = 'serial_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rafeeq',
    maintainer_email='s-nourhan.mahgoub@zewailcity.edu.eg',
    description='Serial bridge between Arduino and ROS2 for Rafeeq robot',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'serial_bridge = serial_node.serial_bridge:main',
        ],
    },
)
