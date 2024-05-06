import os
import datetime as dt
from PIL import Image
import re

import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt


class Trial():
    dim = 2
    '''
    self.frames (np.array): first index is frame, second & third indices frame dimensions (x, y)
    self.batch_df (pd.DataFrame): output of tp.batch
    self.link_df (pd.DataFrame): output of self.link
    '''
    def __init__(self, frames_dir_path, microns_per_pixel):
        '''
        Args:
            frames_dir_path (os.path): Path to folder containing frames (.bmp) of recording
        '''
        # Calculate bmp fnames
        # File names prefixed by dirname
        frames = []
        frames_dirname = os.path.basename(frames_dir_path)
        self.trial_name = frames_dirname
        fnames = (f"{frames_dirname} Frame {i+1}.bmp" for i in range(len(os.listdir(frames_dir_path))))
        for f in fnames:
            try:
                frames.append(np.array(Image.open(os.path.join(frames_dir_path, f)).convert('L')))
            except:
                break

        # Get time of each frame from frames.txt file
        frames_path = os.path.join(frames_dir_path, f"{frames_dirname} frames.txt")
        time_map = {'frame':[], 'time (s)':[]}
        first_datetime = None
        with open(frames_path, 'r') as f:
            next_line = int(next(f).replace(f"{frames_dirname} Frame ", '').replace(".bmp\n", ''))
            while next_line:
                time_str = next(f).strip('\n')
                ms_str = next(f).strip('\n')
                if first_datetime is None:
                    first_datetime = dt.datetime.strptime(time_str, "%H:%M:%S %p") + dt.timedelta(milliseconds=int(ms_str))
                time_map['frame'].append(next_line - 1)
                dt_in_s = (dt.datetime.strptime(time_str, "%H:%M:%S %p") + dt.timedelta(milliseconds=int(ms_str)) - first_datetime).total_seconds()
                time_map['time (s)'].append(dt_in_s)
                try:
                    next_line = int(next(f).replace(f"{frames_dirname} Frame ", '').replace(".bmp\n", ''))
                except:
                    break

        self.frame_time_df = pd.DataFrame.from_dict(time_map)

        # Frames to 3d np array
        self.frames = np.array(frames)
        self.microns_per_pixel = microns_per_pixel

        # Dataframes (to be initialized with self.batch() and self.link()
        self.batch_df = None
        self.link_df = None

    #################### Locating/Tracking Functions ####################
    
    def locate(self, frame_num, diameter, minmass) -> None:
        '''
        Helper function to find optimal parameters. Locates and tracks particles in [frame_num] and plots output
        '''
        f = tp.locate(raw_image=self.frames[frame_num], diameter=diameter, minmass=minmass)
        ax = tp.annotate(f, self.frames[frame_num])
            

    def batch(self, diameter, output=None, meta=None, processes='auto', after_locate=None, **kwargs) -> None:
        '''
        Calls tp.batch on self.frames, stores result in self.batch_df
        Converts to microns at this step
        '''
        self.batch_df = tp.batch(frames=self.frames, diameter=diameter, output=output, meta=meta, processes=processes, after_locate=after_locate, **kwargs)

    def link(self, search_range, pos_columns=None, t_column='frame', memory=0, predictor=None, adaptive_stop=None, adaptive_step=0.95, neighbor_strategy=None, link_strategy=None, dist_func=None, to_eucl=None) -> None:
        '''
        Calls tp.link on self.batch_df, stores result in self.link_df
        *NOTE* after this, units are in microns
        '''
        try:
            self.link_df = tp.link(
                self.batch_df,
                search_range,
                pos_columns=pos_columns,
                t_column=t_column,
                memory=memory,
                predictor=predictor,
                adaptive_stop=adaptive_stop,
                adaptive_step=adaptive_step,
                neighbor_strategy=neighbor_strategy,
                link_strategy=link_strategy,
                dist_func=dist_func,
                to_eucl=to_eucl
            )
            self.link_df = self.link_df.merge(self.frame_time_df, on='frame', how='inner')
            self.link_df['x'] *= self.microns_per_pixel
            self.link_df['y'] *= self.microns_per_pixel
            
        except TypeError as e:
            print("batch_df not initialized! Call self.batch() first!")

    def get_particle(self, particle_num) -> pd.DataFrame:
        '''
        Gets trajectory data for particle [particle_num]
        '''
        return self.link_df[self.link_df['particle'] == particle_num].copy()

    #################### Calculation Functions ####################
    
    def get_squared_disp_single(self, particle_num) -> pd.DataFrame:
        '''
        Calculate squared displacement for single particle
        '''
        part_df = self.get_particle(particle_num)
        
        # Get x, y, and time
        cols = part_df[['x', 'y', 'time (s)']].copy()

        # Remove initial position
        cols['x'] -= cols.iloc[0]['x']
        cols['y'] -= cols.iloc[0]['y']

        # Calculate squared disp
        cols['squared_disp'] = cols['x']**2 + cols['y']**2

        return cols

    def get_squared_disp(self) -> {int:pd.DataFrame}:
        '''
        Calculate squared displacement for all particles, return in dict keyed by particle number
        '''
        squared_disp_dict = {}
        particles = self.link_df['particle'].unique()

        for particle_num in particles:
            squared_disp_dict[particle_num] = self.get_squared_disp_single(particle_num)

        return squared_disp_dict
        
    def get_dsquared_single(self, particle_num) -> pd.DataFrame:
        '''
        Calculate delta squared displacement for a single particle
        '''
        part_df = self.get_particle(particle_num)
        
        # Get x, y, and time and calculate diff (and remove nan), then rename columns
        cols = part_df[['x', 'y', 'time (s)']].copy().diff().dropna().rename(columns={'x':'delta x', 'y':'delta y', 'time (s)':'delta time (s)'})
        
        # Calculate displacement squared
        cols['dsquared'] = cols['delta x']**2 + cols['delta y']**2

        # Drop first col (NaN)

        return cols

    def get_dsquared(self) -> {int:pd.DataFrame} :
        '''
        Calculate delta squared displacement for each particle and return dictionary, keyed by particlenum
        '''
        dsquared_dict = {}
        particles = self.link_df['particle'].unique()

        for particle_num in particles:
            dsquared_dict[particle_num] = self.get_dsquared_single(particle_num)

        return dsquared_dict

    def get_diffusion_coefficient_single(self, particle_num) -> (float, float):
        '''
        Calculates diffusion coefficient of single particle. Returns (value, error) in m^2/s
        '''
        dsquared_df = self.get_dsquared_single(particle_num)
        
        # Calculate tau as average time between frames
        tau = dsquared_df['delta time (s)'].mean()

        # Convert microns to meters
        dsquared_df['dsquared'] *= 10**-12

        # Calculate diff coeff
        diff_coeff = dsquared_df['dsquared'].mean() / (2 * tau * self.dim)

        # Calculate error in measurement
        # Note that our N value is 1 less than the actual number of frames since we are using difference
        err_diff_coeff = dsquared_df['dsquared'].std() / (2 * tau * self.dim * np.sqrt(dsquared_df.index.size))
        
        
        return (diff_coeff, err_diff_coeff)

    def get_diffusion_coefficients(self):
        '''
        Calculates diffusion coefficient for all particles
        '''
        coeffs = []
        particles = self.link_df['particle'].unique()

        for p in particles:
            coeffs.append(self.get_diffusion_coefficient_single(p))
        return coeffs
    
    #################### Plotting Functions ####################
    
    def plot_traj_single(self, particle_num) -> None:
        '''
        Plots trajectory of single particle
        '''
        part_df = self.get_particle(particle_num)

        # Get x and y positions (pixels)
        x = part_df['x']
        y = part_df['y']

        plt.plot(x, y)

    def plot_traj(self) -> None:
        '''
        Plots trajectories of this sample
        '''
        try:
            particles = self.link_df['particle'].unique()
            for p in particles:
                self.plot_traj_single(p)
            # tp.plot_traj(self.link_df)
        except TypeError as e:
            print("link_df not initialized! Call self.link() first!")

    def plot_squared_disp_single(self, particle_num) -> None:
        '''
        Plots squared displacement for single particle
        '''
        df = self.get_squared_disp_single(particle_num)
        
        # Plot
        plt.scatter(df['time (s)'], df['squared_disp'])
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement Squared (microns)')

    def plot_squared_disp(self) -> None:
        '''
        Plots squared displacement for all particles
        '''
        squared_disp_dict = self.get_squared_disp()

        for key, val in squared_disp_dict.items():
            plt.plot(val['time (s)'], val['squared_disp'], label=key)
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement Squared (microns)')

    def plot_diffusion_coefficients(self) -> None:
        '''
        Plots diffusion coefficients with errorbars for all particles
        '''
        # Get particle numbers
        part_nums = self.link_df['particle'].unique()

        # Get diff coeffs
        diff_coeffs = {}
        err_diff_coeffs = {}
        for part_num in part_nums:
            coeff, err = self.get_diffusion_coefficient_single(part_num)
            diff_coeffs[part_num] = coeff
            err_diff_coeffs[part_num] = err
        
        # Plot
        plt.errorbar(diff_coeffs.keys(), diff_coeffs.values(), yerr=list(err_diff_coeffs.values()), capsize=5, ls='none', fmt='o')
        plt.xlabel('Particle')
        plt.ylabel('Diffusion Coefficient')
        plt.xticks(part_nums)





class Experiment():
    '''
    An experiment is a collection of trials. One experiment folder should contain many trial folders
    '''
    def __init__(self, experiment_dir_path, microns_per_pixel):

        self.trials = {}
        trial_dirnames = os.listdir(experiment_dir_path)
        for trial_dirname in trial_dirnames:
            trial_path = os.path.join(experiment_dir_path, trial_dirname)
            if os.path.isdir(trial_path):
                self.trials[trial_dirname] = {
                    'trial': Trial(trial_path, microns_per_pixel),
                    'diameter': None,
                    'minmass': None,
                    'search_range': 10
                }

    #################### Locating/Tracking Functions ####################
    
    def locate(self, trial_name, frame_num, diameter, minmass):
        '''
        Calls locate on specified trial, and sets the parameters stored for the trial to passed in parameters
        '''
        # Get py obj of trial
        trial = self.trials[trial_name]['trial']

        # Call locate func
        trial.locate(frame_num=frame_num, diameter=diameter, minmass=minmass)

        # Update dict parameters
        self.trials[trial_name]['diameter'] = diameter
        self.trials[trial_name]['minmass'] = minmass

    def batch_all(self):
        '''
        Calls batch on all trials. Relevant parameters should be filled before calling
        '''
        for key, val in self.trials.items():
            trial = val['trial']
            if val['diameter'] is None or val['minmass'] is None:
                print(f"{key} parameters not set!")
                return
            print(f"Calling batch on {key} with d={val['diameter']}, minmass={val['minmass']}")
            trial.batch(diameter=val['diameter'], minmass=val['minmass'])

        print("Success!")

    def link_all(self):
        '''
        Calls link on all trials
        '''
        for key, val in self.trials.items():
            trial = val['trial']
            print(f"Calling link_df on {key} with search_range={val['search_range']}")
            trial.link(val['search_range'])

        print("Success!")

    #################### Plotting Functions ####################
    def get_diffusion_coefficient_trial(self, trial_name):
        trial = self.trials[trial_name]['trial']
        return trial.get_diffusion_coefficients()

    #################### Plotting Functions ####################
    
    def plot_traj_trial(self, trial_name):
        trial = self.trials[trial_name]['trial']
        trial.plot_traj()

    def plot_traj_all(self):
        for key, val in self.trials.items():
            trial = val['trial']
            trial.plot_traj()

    
    