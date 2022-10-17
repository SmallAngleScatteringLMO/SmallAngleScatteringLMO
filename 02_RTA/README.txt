This code is part of the SmallAngleScatteringLMO repository,
please read the license file downloading or using this code.
Attribution is required.



This folder contains the code used to calculate the RTA MR 
for the LMO Fermi surfaces within an isotropic-tau approximation.

Please execute 'tests.py' file and make sure all is OK.
Then execute 'execute.py' in the command line or your favourite IDE.
>> Note that the output is already in the "/output" folder
    but showing the results currently requires re-executing.
>> These results files are used for the final figure's RTA results

To show the c-axis warping on the Fermi surface, execute 'c_axis_warping_show.py'
To calculate the influence of c-axis warping execute 'c_axis_calculate.py'
>> Note that the output is already in the "/output c-axis" folder
To show the results of the c-axis calculation execute 'c_axis_visualize_results'

The 'core.py' file contains the code tested with 'tests.py' that really performs
    the calcuation. The other code files listed above are scripts that
    can be edited to change settings and store or show the results.
