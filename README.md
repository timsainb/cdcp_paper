Context dependant categorical perception in songbirds [CODE FOR PAPER]
==============================

UPDATED: Jan 23 2022

Behavior and Ephys analyses for context-dependent categorical perceptual decision making experiment in songbirds.




### Running this analysis
- This analysis is self-contained within this folder structure. Data are kept in the data folder. Raw data amount to over 100TB of continuous neural recordings. 
- Notebooks are numbered in the order that they need to be run and labelled with what they do. 


Project Organization
------------

    ├── LICENSE
    ├── README.md                          <- The top-level README for developers using this project.
    ├── data                               <- all data
    ├── notebooks-clean                    <- All analysis notebooks (a curated version of just the relevant notebooks for the paper)
    ├── notebooks                          <- All analysis notebooks (disorganized)
    ├── __init__.py                        <- Makes src a Python module
    ├── requirements.txt                   <- The requirements file for reproducing the analysis environment
    ├── setup.py                           <- makes project pip installable (pip install -e .) so src can be imported
    ├── cdcp                               <- python package contents needed to run notebooks


--------

## Analysis pipeline
The spikesorting pipeline relies on kilosort and spikeinterface. Data are recorded in ~12 hour blocks continuously for weeks to months and sorted offline. An overview of the pipeline is drawn below. Each box roughly corresponds to a notebook in `notebooks_clean`.

![spikesorting pipeline](assets/spikesorting-pipeline-cdcp-jan-21.svg)




## Spikesorting output
For each bird, we produce the datasets listed below. These 

### Recording summary dataframe
Information about each raw recording session

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dat_file</th>
      <th>sample_rate</th>
      <th>channels</th>
      <th>n_samples</th>
      <th>experiment_num</th>
      <th>recording_num</th>
      <th>date_str</th>
      <th>num_channels_total</th>
      <th>ADC_data_channels</th>
      <th>site</th>
      <th>AP</th>
      <th>ML</th>
      <th>depth</th>
      <th>hemisphere</th>
      <th>probes</th>
      <th>bad_channels</th>
      <th>recording_ID</th>
      <th>datetime</th>
      <th>n_channels</th>
      <th>dat_size_gb</th>
      <th>n_ttl_events</th>
      <th>n_trials</th>
      <th>n_playbacks</th>
      <th>n_response</th>
      <th>n_punish</th>
      <th>n_reward</th>
      <th>n_hours</th>
      <th>recording_id</th>
      <th>site_loc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/mnt/sphere/RawData/Samamba/ephys/B1597/2021-0...</td>
      <td>30000</td>
      <td>[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,...</td>
      <td>827683072</td>
      <td>1</td>
      <td>1</td>
      <td>2021-05-13_14-35-47_1150</td>
      <td>72</td>
      <td>[64, 65, 66, 67, 68, 69, 70, 71]</td>
      <td>5</td>
      <td>2420</td>
      <td>2240</td>
      <td>1150</td>
      <td>R</td>
      <td>[[neuronexus, Buzsaki64]]</td>
      <td>[]</td>
      <td>92</td>
      <td>2021-05-13 14:35:47.115</td>
      <td>64</td>
      <td>119.186362</td>
      <td>12532</td>
      <td>8</td>
      <td>6217</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7.663732</td>
      <td>exp1_rec1_dat2021-05-13_14-35-47_1150</td>
      <td>2420_2240_1150_R</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/mnt/sphere/RawData/Samamba/ephys/B1597/2021-0...</td>
      <td>30000</td>
      <td>[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,...</td>
      <td>64557056</td>
      <td>1</td>
      <td>1</td>
      <td>2021-05-13_22-15-40_1150</td>
      <td>72</td>
      <td>[64, 65, 66, 67, 68, 69, 70, 71]</td>
      <td>5</td>
      <td>2420</td>
      <td>2240</td>
      <td>1150</td>
      <td>R</td>
      <td>[[neuronexus, Buzsaki64]]</td>
      <td>[]</td>
      <td>116</td>
      <td>2021-05-13 22:15:40.115</td>
      <td>64</td>
      <td>9.296216</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.597751</td>
      <td>exp1_rec1_dat2021-05-13_22-15-40_1150</td>
      <td>2420_2240_1150_R</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/mnt/sphere/RawData/Samamba/ephys/B1597/2021-0...</td>
      <td>30000</td>
      <td>[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,...</td>
      <td>709996049</td>
      <td>1</td>
      <td>1</td>
      <td>2021-05-14_10-03-04_1150</td>
      <td>72</td>
      <td>[64, 65, 66, 67, 68, 69, 70, 71]</td>
      <td>5</td>
      <td>2420</td>
      <td>2240</td>
      <td>1150</td>
      <td>R</td>
      <td>[[neuronexus, Buzsaki64]]</td>
      <td>[]</td>
      <td>15</td>
      <td>2021-05-14 10:03:04.115</td>
      <td>64</td>
      <td>102.239431</td>
      <td>10</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.574037</td>
      <td>exp1_rec1_dat2021-05-14_10-03-04_1150</td>
      <td>2420_2240_1150_R</td>
    </tr>
  </tbody>
</table>

### Merged units
Which units have been merged across days

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_id</th>
      <th>n_spikes</th>
      <th>n_trials</th>
      <th>n_playbacks</th>
      <th>sort_units</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3059</th>
      <td>3428</td>
      <td>40848147</td>
      <td>27960.0</td>
      <td>136273.0</td>
      <td>[(231, exp1_rec1_dat2021-05-18_09-10-29_1150),...</td>
    </tr>
    <tr>
      <th>4318</th>
      <td>4957</td>
      <td>18987348</td>
      <td>17933.0</td>
      <td>94700.0</td>
      <td>[(88, exp1_rec1_dat2021-05-21_22-32-02_1150), ...</td>
    </tr>
    <tr>
      <th>7237</th>
      <td>8542</td>
      <td>5758671</td>
      <td>18521.0</td>
      <td>77227.0</td>
      <td>[(18, exp1_rec1_dat2021-05-28_22-06-52_1150), ...</td>
    </tr>
  </tbody>
</table>


### Recording features
Physical features of each unit

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>template</th>
      <th>presence_ratio</th>
      <th>isi_violations_rate</th>
      <th>max_channel</th>
      <th>center_of_mass_x</th>
      <th>center_of_mass_y</th>
      <th>spike_amplitude</th>
      <th>amp_channel_0</th>
      <th>amp_channel_1</th>
      <th>amp_channel_2</th>
      <th>amp_channel_3</th>
      <th>amp_channel_4</th>
      <th>amp_channel_5</th>
      <th>amp_channel_6</th>
      <th>amp_channel_7</th>
      <th>amp_channel_8</th>
      <th>amp_channel_9</th>
      <th>amp_channel_10</th>
      <th>amp_channel_11</th>
      <th>amp_channel_12</th>
      <th>amp_channel_13</th>
      <th>amp_channel_14</th>
      <th>amp_channel_15</th>
      <th>amp_channel_16</th>
      <th>amp_channel_17</th>
      <th>amp_channel_18</th>
      <th>amp_channel_19</th>
      <th>amp_channel_20</th>
      <th>amp_channel_21</th>
      <th>amp_channel_22</th>
      <th>amp_channel_23</th>
      <th>amp_channel_24</th>
      <th>amp_channel_25</th>
      <th>amp_channel_26</th>
      <th>amp_channel_27</th>
      <th>amp_channel_28</th>
      <th>amp_channel_29</th>
      <th>amp_channel_30</th>
      <th>amp_channel_31</th>
      <th>amp_channel_32</th>
      <th>amp_channel_33</th>
      <th>amp_channel_34</th>
      <th>amp_channel_35</th>
      <th>amp_channel_36</th>
      <th>amp_channel_37</th>
      <th>amp_channel_38</th>
      <th>amp_channel_39</th>
      <th>amp_channel_40</th>
      <th>amp_channel_41</th>
      <th>amp_channel_42</th>
      <th>amp_channel_43</th>
      <th>amp_channel_44</th>
      <th>amp_channel_45</th>
      <th>amp_channel_46</th>
      <th>amp_channel_47</th>
      <th>amp_channel_48</th>
      <th>amp_channel_49</th>
      <th>amp_channel_50</th>
      <th>amp_channel_51</th>
      <th>amp_channel_52</th>
      <th>amp_channel_53</th>
      <th>amp_channel_54</th>
      <th>amp_channel_55</th>
      <th>amp_channel_56</th>
      <th>amp_channel_57</th>
      <th>amp_channel_58</th>
      <th>amp_channel_59</th>
      <th>amp_channel_60</th>
      <th>amp_channel_61</th>
      <th>amp_channel_62</th>
      <th>amp_channel_63</th>
      <th>snrs</th>
      <th>amplitude_cutoff</th>
      <th>best_channel_0</th>
      <th>best_channel_1</th>
      <th>best_channel_2</th>
      <th>best_channel_3</th>
      <th>best_channel_4</th>
      <th>best_channel_5</th>
      <th>best_channel_6</th>
      <th>best_channel_7</th>
      <th>best_channel_8</th>
      <th>best_channel_9</th>
      <th>isi_violations_count</th>
      <th>n_spikes</th>
      <th>recording_id</th>
      <th>datetime</th>
      <th>unit</th>
      <th>n_hours</th>
      <th>spike_rate</th>
      <th>good_unit</th>
      <th>z_score_template</th>
      <th>median_relative_channel_max</th>
      <th>recording_unit</th>
      <th>cluster_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[[21.0, 4.0, 22.0, 0.0, 39.0, 13.0, -11.0, -10...</td>
      <td>1.0</td>
      <td>0.009460</td>
      <td>0</td>
      <td>66.835239</td>
      <td>24.625780</td>
      <td>-1239.0</td>
      <td>1239.0</td>
      <td>140.0</td>
      <td>43.0</td>
      <td>49.0</td>
      <td>48.0</td>
      <td>41.0</td>
      <td>29.0</td>
      <td>174.0</td>
      <td>35.0</td>
      <td>34.0</td>
      <td>36.0</td>
      <td>31.0</td>
      <td>26.0</td>
      <td>34.0</td>
      <td>37.0</td>
      <td>43.0</td>
      <td>53.0</td>
      <td>87.0</td>
      <td>35.0</td>
      <td>38.0</td>
      <td>35.0</td>
      <td>22.0</td>
      <td>37.0</td>
      <td>27.0</td>
      <td>39.0</td>
      <td>40.0</td>
      <td>29.0</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>47.0</td>
      <td>42.0</td>
      <td>36.0</td>
      <td>39.0</td>
      <td>36.0</td>
      <td>19.0</td>
      <td>23.0</td>
      <td>24.0</td>
      <td>29.0</td>
      <td>37.0</td>
      <td>32.0</td>
      <td>31.0</td>
      <td>29.0</td>
      <td>26.0</td>
      <td>37.0</td>
      <td>28.0</td>
      <td>25.0</td>
      <td>26.0</td>
      <td>19.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>16.0</td>
      <td>22.0</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>44.0</td>
      <td>20.0</td>
      <td>24.0</td>
      <td>32.0</td>
      <td>36.0</td>
      <td>43.0</td>
      <td>24.0</td>
      <td>28.0</td>
      <td>35.0</td>
      <td>20.0</td>
      <td>21.529388</td>
      <td>0.010351</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>17</td>
      <td>16</td>
      <td>3</td>
      <td>4</td>
      <td>29</td>
      <td>54</td>
      <td>2</td>
      <td>261</td>
      <td>78741</td>
      <td>exp1_rec1_dat2021-05-13_14-35-47_1150</td>
      <td>2021-05-13 14:35:47.115</td>
      <td>0</td>
      <td>7.663732</td>
      <td>2.854027</td>
      <td>True</td>
      <td>[[0.40623679964006376, -0.020724543118155827, ...</td>
      <td>0.027476</td>
      <td>exp1_rec1_dat2021-05-13_14-35-47_1150_0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[[1.0, 5.0, -2.0, -12.0, -9.0, -10.0, -9.0, -2...</td>
      <td>1.0</td>
      <td>0.158249</td>
      <td>0</td>
      <td>196.166667</td>
      <td>45.405405</td>
      <td>-350.0</td>
      <td>366.0</td>
      <td>121.0</td>
      <td>34.0</td>
      <td>42.0</td>
      <td>35.0</td>
      <td>24.0</td>
      <td>27.0</td>
      <td>106.0</td>
      <td>48.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>27.0</td>
      <td>23.0</td>
      <td>31.0</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>34.0</td>
      <td>28.0</td>
      <td>30.0</td>
      <td>23.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>32.0</td>
      <td>23.0</td>
      <td>34.0</td>
      <td>29.0</td>
      <td>26.0</td>
      <td>38.0</td>
      <td>42.0</td>
      <td>37.0</td>
      <td>22.0</td>
      <td>27.0</td>
      <td>38.0</td>
      <td>33.0</td>
      <td>31.0</td>
      <td>40.0</td>
      <td>40.0</td>
      <td>42.0</td>
      <td>21.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>37.0</td>
      <td>33.0</td>
      <td>27.0</td>
      <td>29.0</td>
      <td>17.0</td>
      <td>41.0</td>
      <td>30.0</td>
      <td>21.0</td>
      <td>31.0</td>
      <td>34.0</td>
      <td>24.0</td>
      <td>31.0</td>
      <td>25.0</td>
      <td>32.0</td>
      <td>25.0</td>
      <td>26.0</td>
      <td>6.081748</td>
      <td>0.010378</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>8</td>
      <td>29</td>
      <td>38</td>
      <td>3</td>
      <td>53</td>
      <td>36</td>
      <td>37</td>
      <td>4366</td>
      <td>182668</td>
      <td>exp1_rec1_dat2021-05-13_14-35-47_1150</td>
      <td>2021-05-13 14:35:47.115</td>
      <td>1</td>
      <td>7.663732</td>
      <td>6.620940</td>
      <td>True</td>
      <td>[[-0.09374838525172778, 0.11372090647815486, -...</td>
      <td>0.088644</td>
      <td>exp1_rec1_dat2021-05-13_14-35-47_1150_1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[[12.0, 13.0, 21.0, 37.0, 52.0, 21.0, 19.0, 16...</td>
      <td>1.0</td>
      <td>0.035485</td>
      <td>1</td>
      <td>38.383940</td>
      <td>48.996236</td>
      <td>-722.0</td>
      <td>121.0</td>
      <td>722.0</td>
      <td>224.0</td>
      <td>69.0</td>
      <td>38.0</td>
      <td>94.0</td>
      <td>59.0</td>
      <td>161.0</td>
      <td>28.0</td>
      <td>27.0</td>
      <td>24.0</td>
      <td>33.0</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>29.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>43.0</td>
      <td>48.0</td>
      <td>39.0</td>
      <td>24.0</td>
      <td>30.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>43.0</td>
      <td>31.0</td>
      <td>34.0</td>
      <td>47.0</td>
      <td>49.0</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>30.0</td>
      <td>35.0</td>
      <td>31.0</td>
      <td>24.0</td>
      <td>25.0</td>
      <td>31.0</td>
      <td>29.0</td>
      <td>31.0</td>
      <td>38.0</td>
      <td>25.0</td>
      <td>21.0</td>
      <td>26.0</td>
      <td>32.0</td>
      <td>23.0</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>26.0</td>
      <td>16.0</td>
      <td>34.0</td>
      <td>29.0</td>
      <td>15.0</td>
      <td>22.0</td>
      <td>37.0</td>
      <td>26.0</td>
      <td>36.0</td>
      <td>34.0</td>
      <td>28.0</td>
      <td>42.0</td>
      <td>42.0</td>
      <td>12.510785</td>
      <td>0.010351</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>29</td>
      <td>19</td>
      <td>28</td>
      <td>979</td>
      <td>142284</td>
      <td>exp1_rec1_dat2021-05-13_14-35-47_1150</td>
      <td>2021-05-13 14:35:47.115</td>
      <td>2</td>
      <td>7.663732</td>
      <td>5.157191</td>
      <td>True</td>
      <td>[[0.29079075755345496, 0.324157719653696, 0.59...</td>
      <td>0.045106</td>
      <td>exp1_rec1_dat2021-05-13_14-35-47_1150_2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>

### Trial-aligned spikes
These dataframes actual spike timing in relation to trials. 

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stim</th>
      <th>trial_id</th>
      <th>frame_begin</th>
      <th>correct</th>
      <th>response</th>
      <th>punish</th>
      <th>reward</th>
      <th>stim_length</th>
      <th>unit</th>
      <th>spike_times</th>
      <th>passive</th>
      <th>n_spikes</th>
      <th>recording_id</th>
      <th>cue</th>
      <th>interp</th>
      <th>interp_point</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CR1_AF_058</td>
      <td>2021-05-18 09:10:30.656118</td>
      <td>34016</td>
      <td>False</td>
      <td>left</td>
      <td>True</td>
      <td>False</td>
      <td>1.991033</td>
      <td>231</td>
      <td>[1.4212, 1.9128666666666667]</td>
      <td>False</td>
      <td>2</td>
      <td>exp1_rec1_dat2021-05-18_09-10-29_1150</td>
      <td>CR1</td>
      <td>AF</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CR1_AF_058</td>
      <td>2021-05-18 09:10:49.539861</td>
      <td>600743</td>
      <td>True</td>
      <td>right</td>
      <td>False</td>
      <td>False</td>
      <td>1.991033</td>
      <td>231</td>
      <td>[0.8069666666666667, 1.4312666666666667, 1.8599]</td>
      <td>False</td>
      <td>3</td>
      <td>exp1_rec1_dat2021-05-18_09-10-29_1150</td>
      <td>CR1</td>
      <td>AF</td>
      <td>58</td>
    </tr>
    <tr>
      <th>0</th>
      <td>CR1_BE_063</td>
      <td>2021-05-18 09:10:55.768172</td>
      <td>787625</td>
      <td>True</td>
      <td>right</td>
      <td>False</td>
      <td>False</td>
      <td>1.991067</td>
      <td>231</td>
      <td>[1.9591333333333334]</td>
      <td>False</td>
      <td>1</td>
      <td>exp1_rec1_dat2021-05-18_09-10-29_1150</td>
      <td>CR1</td>
      <td>BE</td>
      <td>63</td>
    </tr>
  </tbody>
</table>


### Behavior responses
These dataframes have all of the bird's behavioral responses, including chronic recording trials. 

<table border="1" class="dataframe">
  <thead>
    <tr style="undefined:undefined">
      <th></th>
      <th>session</th>
      <th>index</th>
      <th>type_</th>
      <th>stimulus</th>
      <th>class_</th>
      <th>response</th>
      <th>correct</th>
      <th>rt</th>
      <th>reward</th>
      <th>punish</th>
      <th>cue_class</th>
      <th>cue_id</th>
      <th>cue_prob</th>
      <th>num_stims</th>
      <th>flip_cues</th>
      <th>binary_choice</th>
      <th>cueing</th>
      <th>left_stim</th>
      <th>right_stim</th>
      <th>interpolation_point</th>
      <th>prob_cue</th>
      <th>prob_cued_no_cue</th>
      <th>data_file</th>
      <th>response_bool</th>
      <th>interpolation</th>
      <th>pos_bin</th>
      <th>cue_direction</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-02-28 15:31:37.133368</th>
      <td>1</td>
      <td>5</td>
      <td>normal</td>
      <td>/home/bird/opdat/ts_cue_prob_multicue_stims/AE...</td>
      <td>R</td>
      <td>L</td>
      <td>False</td>
      <td>3.326432</td>
      <td>False</td>
      <td>True</td>
      <td>NC</td>
      <td>NC</td>
      <td>0.5</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>A</td>
      <td>E</td>
      <td>127</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>/mnt/cube/RawData/Magpi/B1174/B1174_trialdata_...</td>
      <td>0</td>
      <td>AE</td>
      <td>24</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2019-02-28 15:32:03.886668</th>
      <td>1</td>
      <td>6</td>
      <td>normal</td>
      <td>/home/bird/opdat/ts_cue_prob_multicue_stims/AE...</td>
      <td>L</td>
      <td>R</td>
      <td>False</td>
      <td>4.199297</td>
      <td>False</td>
      <td>True</td>
      <td>NC</td>
      <td>NC</td>
      <td>0.5</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>A</td>
      <td>E</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>/mnt/cube/RawData/Magpi/B1174/B1174_trialdata_...</td>
      <td>1</td>
      <td>AE</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2019-02-28 15:32:34.116591</th>
      <td>1</td>
      <td>9</td>
      <td>normal</td>
      <td>/home/bird/opdat/ts_cue_prob_multicue_stims/AE...</td>
      <td>R</td>
      <td>R</td>
      <td>True</td>
      <td>0.761013</td>
      <td>True</td>
      <td>False</td>
      <td>NC</td>
      <td>NC</td>
      <td>0.5</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>A</td>
      <td>E</td>
      <td>127</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>/mnt/cube/RawData/Magpi/B1174/B1174_trialdata_...</td>
      <td>1</td>
      <td>AE</td>
      <td>24</td>
      <td>N</td>
    </tr>
  </tbody>
</table>

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
