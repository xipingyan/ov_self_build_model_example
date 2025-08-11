source ../../python-env/bin/activate
source /mnt/xiping/openvino/build/install/setupvars.sh

# onetrace --chrome-call-logging --chrome-device-timeline
python test_custom_op_2_outputs.py