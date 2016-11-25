# EEGpy
October 2015

EEGpy is a system for the analysis of EEG data.  The system features all the necessary preprocessing of the input EEG signals, including artifact detection and removal, and is focused on the analysis of coherence between the different EEG electrodes, i.e. the different brain regions that they correspond to. We see the presented system as a good foundation for further extension as it is applied to different areas of EEG analysis. EEGpy is free software and we hope that it will be useful to the research community.

To run EEGpy, clone it from `git`, place your EEG signals in `eegs/` and execute `eeg_load_and_clean_data.py`, `eeg_process_data_loop.py`, and `eeg_coherence_master.py` in a sequence. For convenience a single EEG recording is included in the repo.

At the moment EEGpy supports reading EEGs stored in the EDF file format through the `edfplus.py` module which is a part of Boris Reuderink's `eegtools` set of libraries that can be found [here.](https://github.com/breuderink/eegtools)

For ploting the EEG topographic maps, EEGpy superimposes the plots on the figure `21ch_eeg.png` by トマトン124 (talk) - Own work, Public Domain, which can be found [here.](https://commons.wikimedia.org/w/index.php?curid=10489987)

EEGpy is described in the paper:

Gerazov B. and S. Markovska-Simoska, “EEGpy: A system for the analysis of coherence in EEG data used in the assessment of ADHD,” ETAI, Ohrid, Macedonia, Sep 22-24, 2016.

```
@inproceedings{Gerazov2016,
    author = {Gerazov, Branislav and Markovska-Simoska, Silvana},
    title = {EEGpy: A system for the analysis of coherence in EEG data used in the assessment of ADHD},
    booktitle = {Proceedings of ETAI},
    location = {Struga, Macedonia},
    month = {Sep},
    year = {2016}
}
```

Branislav Gerazov

[DIPteam](http://dipteam.feit.ukim.edu.mk/)

[Faculty of Electrical Engineering and Information Technologies](http://feit.ukim.edu.mk)

Ss Cyril and Methodius University of Skopje, Macedonia
