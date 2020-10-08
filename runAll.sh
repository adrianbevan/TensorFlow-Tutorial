#    runAll.sh - a simple shell script to run the hyperparameter optimisation scripts for
#                the MLP example NN.py.
#    Copyright (C) 2020 Adrian Bevan, Queen Mary University of London
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

echo ""
echo "Launch the training scans for the MNIST MLP example"
echo ""
echo "BEFORE running this script, make sure the log file directory exists"
echo ""
python ValidationSplitNN.py >& log/ValidationSplitNN.log &
python BatchSizeNN.py       >& log/BatchSizeNN.log &
python LeakyReluScanNN.py   >& log/LeakyReluScanNN.log &
python DropoutNN.py         >& log/DropoutNN.log &
echo ""
echo "ValidationSplitNN, BatchSizeNN, LeakyReluScanNN, DropoutNN are all running in parallel"
echo ""

