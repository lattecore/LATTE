1. Generate model configurations.

   ```
   python generate_model_configs.py
   ```

2. Generate the corresponding exhaustive testing results and heuristic results.

   1. Run the generate_ground_truth.cpp by using nvcc in my environment (you may change the following commands to match your environment):

      ```
      nvcc .\generate_ground_truth.cpp -o output -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64" -lcudnn -lcublas -lcudart
      ```

   2. Run the generate_heuristic_result.cpp by using nvcc in my environment (you may change the following commands to match your environment):

      ```
      nvcc .\generate_heuristic_result.cpp -o output -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64" -lcudnn -lcublas -lcudart
      ```

3. Compare the exhaustive testing results and heuristic results.

   ```
   python compare_get_find.py
   ```

4. Layer Algorithm Selector

   ```
   python Layer_Algorithm_Selector.py
   ```

5. Training Time Estimator

   ```
   python Training_Time_Estimator.py
   ```

   