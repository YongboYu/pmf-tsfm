- "takeaway" -> "signals"
  - complexity and overfitting (shortage of training samples)
  - common patterns in process time series data

- based on a valid, solid and extensive baselines (benchmarking) -> TSFMs have really exciting performance

- the core is a time series forecasting problem but for process data may share this general patterns -> so this work is meaningful and more importantly inspirable for our commuinty
- gross timing profile -> foundation models save a lot of training effort
- more intrinsic thinking: process as a sequence may somehow encode as tokens in LLMs or some dedicated foundation models tailored to process nature
- in another sense: foundation models need much larger corpus for training
- artifacts: GitHub, featured with local (mps), linux (cuda), HPC
- In many operational contexts, however, anticipating structural process change is at least as
  important as understanding past behaviour. Compliance teams need to know whether a new
  regulation will alter the dominant routing of cases before it takes effect. Capacity planners must
  estimate which activities will be co-active in the coming weeks. Anomaly-detection systems
  benefit from knowing the expected model against which deviations should be measured.
- rough timing profile: 3 zero-shot vs 3 lora-tuning vs 3 full-tuning vs 1 XGBoost
- Manim library with AI to draw pretty math figures
- PPM: predict at the moment of each new event observed -> time granularity, PMF: fix granualarity to make modeling the evolument possible (again, daily as normal and practical instead of random equal slicing)
- Demo / artifact: input log output process model (PMF)
- maybe even put it in an agent space that people can use natural language to download and inference
