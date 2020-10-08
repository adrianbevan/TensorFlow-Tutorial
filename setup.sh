#
# source setup.sh to ensure that the relevant directories are setup for the tutorial
# This script assumes that the user is working with a UNIX/LINUX derived operating
# system running bash.
#
echo "....... Sourcing the TensorFlow-Tutorial setup.sh script"
echo "---------------------------------------------------------------"
echo ""
echo "     setup.sh Copyright (C) 2020 Adrian Bevan"
echo ""
echo " This program comes with ABSOLUTELY NO WARRANTY."
echo " This is free software, and you are welcome to redistribute it"
echo " under certain conditions."
echo ""
echo "---------------------------------------------------------------"

echo ""
echo "Make directories required for the tutorial output"
echo ""
if [ ! -d log ]; then
  echo "  Making the directory ./fig"
  mkdir fig
else
  echo "  The directory ./fig already exists"
fi

if [ ! -d log ]; then
  echo "  Making the directory ./log"
  mkdir log
else
  echo "  The directory ./log already exists"
fi
echo ""


echo "Please run the following command to check that these now exist"
echo ""
echo "   ls "
echo ""
echo "In addition to the python files for this tutorial you should now see two new directories:"
echo ""
echo "  ./fig    - used by the scripts to save output plots generated during model training"
echo "  ./log    - used by the runAll.sh script to save log files for the optimisation scripts:"
echo ""
echo "              BatchSizeNN.py       - scan through different batch sizes to study loss"
echo "                                     and accuracy performance of training as a function"
echo "                                     of this hyperparameter"
echo ""
echo "              DropoutNN.py         - scan through different dropout rates to study loss"
echo "                                     and accuracy performance of training as a function"
echo "                                     of this hyperparameter"
echo ""
echo "              LeakyReluScanNN.py   - scan through different leaky rates to study loss"
echo "                                     and accuracy performance of training as a function"
echo "                                     of this hyperparameter"
echo ""
echo "              ValidationSplitNN.py - scan through different validation splits to study loss"
echo "                                     and accuracy performance of training as a function"
echo "                                     of this hyperparameter"
echo ""
