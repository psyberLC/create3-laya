from glob import glob

from setuptools import find_packages, setup

package_name = 'create3_laya'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='create3-laya maintainer',
    maintainer_email='lcpsyber@gmail.com',
    description='Plain-English tasks for the iRobot Create 3, interpreted by a local Laya model',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'laya_create3 = create3_laya.laya_node:main',
            'laya_task = create3_laya.task_cli:main',
            'clock_relay = create3_laya.clock_relay:main',
        ],
    },
)
