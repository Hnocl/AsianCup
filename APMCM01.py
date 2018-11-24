import os
import numpy as np
import time
import math
import csv
from sklearn import linear_model

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

input_dir = "D:\\大学数学\\数学建模\\亚太\\2018 APMCM Problems\\2018 APMCM Problem A\\Annex 2 raw data"
filewalk = os.walk(input_dir)
for a,b,c in filewalk:
    root = a
    files = c
paths = [root+"\\"+i for i in files]

csv_file = csv.reader(open("D:\\大学数学\\数学建模\\forMCM\\亚太\\Annex 1 Basic Data of Elderly People.csv",'r'))
people_info = 0
for each_line in csv_file:
    #print(len(each_line))
    #print(each_line)
    if people_info == 0:
        people_info = []
        continue
    
    people_info.append(each_line)
    #break
#print(people_info)
#print(files)
class Person:
    def __init__(self,file):
        self.data = []
        name = file.split("\\")[-1]
        name = name[:-4]
        self.name = name
        self.message = []

        for index in range(len(people_info)):
            if people_info[index][1] in self.name:
                self.message = people_info[index]
        if self.message == []:
            print("Warning: No corresponding message found.")
            print("name:",self.name)

        f = open(file,'r')
        for each_line in f:
            line = each_line.split("\t")
            line = line[:-1]
            line = [float(each) for each in line]
            self.data.append(line)
        self.data = np.asarray(self.data)
        self.id = self.data[0]
        self.time_seq = self.data[:,1]
        self.monitor = {}
        for i in range(42):
            self.monitor[i+1] = self.data[:,3*i+2:3*i+5]
    def get_plot_data(self,index):
        res_x = self.data[index][2::3]
        res_y = self.data[index][3::3]
        res_z = self.data[index][4::3]
        return res_x,res_y,res_z

def cover(num):
    return "%05d"%num

def gif_gen(person):
    count = 0
    for index,each_t in enumerate(person.time_seq):
        fig = plt.figure()
        ax = Axes3D(fig)
        #ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs = person.get_plot_data(index)
        ax.scatter(xs, ys, zs, c = 'b')
        
        
        ax.set_zlim(0,2000)
        ax.set_xlim(0,700)
        ax.axis('square')
        #ax.axis('off')
        ax.view_init(elev=3,azim=15)
    
        #plt.show(ax)
        #print(each_t)
        save_path = "亚太\\动作图\\"+person.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+"\\"+cover(count)+".png")
        count += 1

def calc_distance(point1,point2):
    sum = 0.0
    sum += (point1[0]-point2[0])*(point1[0]-point2[0])
    sum += (point1[1]-point2[1])*(point1[1]-point2[1])
    sum += (point1[2]-point2[2])*(point1[2]-point2[2])
    return math.sqrt(sum)


def choose_steplen(data,threshold=5,cut_edge=False):
    section = [data.index(i) for i in data if i < threshold]
    #print(section)

    total_subsection = []
    subsection = []
    #print(section)
    for index in range(section[0],section[-1]+1):
        #print(index)
        if index not in section:
            if subsection == []:
                continue
            total_subsection.append(subsection)
            subsection = []
        else:
            subsection.append(index)
    if subsection != []:
        total_subsection.append(subsection)
    #print(total_subsection)

    res = []
    for each_sub in total_subsection:
        for index in range(len(each_sub)):
            if index == 0:
                res_ele = index
            elif data[each_sub[index]] < data[each_sub[res_ele]]:
                res_ele = index
        if cut_edge:
            if each_sub[res_ele] == 0 or each_sub[res_ele] == len(data)-1:
                continue
        res.append(each_sub[res_ele])
    #print(res)
    return res

def choose_highest(data):
    section = [data.index(i) for i in data if i >120]
    #print(section)

    total_subsection = []
    subsection = []
    #print(section)
    for index in range(section[0],section[-1]+1):
        #print(index)
        if index not in section:
            if subsection == []:
                continue
            total_subsection.append(subsection)
            subsection = []
        else:
            subsection.append(index)
    if subsection != []:
        total_subsection.append(subsection)
    #print(total_subsection)

    res = []
    for each_sub in total_subsection:
        for index in range(len(each_sub)):
            if index == 0:
                res_ele = index
            elif data[each_sub[index]] > data[each_sub[res_ele]]:
                res_ele = index
        res.append(each_sub[res_ele])
    #print(res)
    return res

def intergration(data):
    res = [0.0]
    for index in range(len(data)):
        res.append(res[-1]+data[index])
    return res

def differential(data):
    res = []
    for index in range(len(data)-1):
        res.append(abs(data[index+1]-data[index]))
    return res

def calc_stepsize(person):
    '''
    Calcaluae the mean step size
    '''
    step_size_csv = []
    plt.cla()
    r_points = [25,5,1,34,3,36]
    l_points = [26,6,2,35,4,37]
    for each_feet in [l_points,r_points]:
        data_sum = np.zeros([len(person.time_seq),3])
        for each_point in each_feet:
            data_seq = person.monitor[each_point]
            data_sum += data_seq
        data_mean = data_sum/6
    
        test = []
        for index in range(len(data_mean)-1):
            delta_move = calc_distance(data_mean[index],data_mean[index+1])
            #print(delta_move)
            test.append(delta_move)
        steps = choose_steplen(test)
        total_lengths = []
        for index in range(len(steps)-1):
            total_lengths.append(calc_distance(data_mean[steps[index]],data_mean[steps[index+1]]))
        
        average_length = np.mean(np.asarray(total_lengths))
        leg = "(left leg)" if each_feet == l_points else "(right leg)"
        step_size_csv.append([leg,average_length])
        print(person.name,leg,":",average_length)
        #raise AssertionError()
        #test = intergration(test)
        #test = differential(test)
        plt.plot(np.asarray([0.01666666667*i for i in range(len(test))]),np.asarray(test))

        #feet_height = []
        #for index in range(len(data_mean)):
        #    feet_height.append(data_mean[index][2])
        #plt.plot(np.asarray([0.01666666667*i for i in range(len(feet_height))]),np.asarray(feet_height))

        #print(test.index(sorted(test)[0]))
    plt.xlabel("Time/s")
    plt.ylabel("Displacement/mm")
    
    plt.grid(linestyle='-.')
    plt.legend(("Left feet","Right feet"))
    #plt.show()
    #save_path = "亚太\\步长曲线\\路程变化\\"+person.name+".png"
    #plt.savefig(save_path)
    return step_size_csv

def calc_mean_points(person,point_list):
    data_sum = np.zeros([len(person.time_seq),3])
    for each_point in point_list:
        data_seq = person.monitor[each_point]
        data_sum += data_seq
    data_mean = data_sum/len(point_list)
    return data_mean

def calc_gcenter(person):
    '''
    Calculate the center points
    '''
    plt.cla()
    trunk_p = [33,13,14,29,16,17,31,30,15,42]
    head_p = [22,40,23,24,41,32]
    l_arm_p = [18]
    r_arm_p = [19]
    l_hand_p = [20,38]
    r_hand_p = [21,39]
    l_knee_p = [9,27]
    r_knee_p = [10,28]
    l_feet_points = [26,6,2,35,4,37]
    r_feet_points = [25,5,1,34,3,36]


    trunk_data_mean = calc_mean_points(person,trunk_p)
    head_data_mean = calc_mean_points(person,head_p)
    l_arm = calc_mean_points(person,l_arm_p)
    r_arm = calc_mean_points(person,r_arm_p)
    l_hand = calc_mean_points(person,l_hand_p)
    r_hand = calc_mean_points(person,r_hand_p)
    l_knee = calc_mean_points(person,l_knee_p)
    r_knee = calc_mean_points(person,r_knee_p)
    l_feet = calc_mean_points(person,l_feet_points)
    r_feet = calc_mean_points(person,r_feet_points)
    

    #raise AssertionError()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(trunk_data_mean[:,0], trunk_data_mean[:,1], trunk_data_mean[:,2], c = 'b')
    ax.scatter(head_data_mean[:,0], head_data_mean[:,1], head_data_mean[:,2], c = 'r')
    ax.scatter(l_arm[:,0], l_arm[:,1], l_arm[:,2], c = 'g')
    ax.scatter(r_arm[:,0], r_arm[:,1], r_arm[:,2], c = 'g')
    ax.scatter(l_hand[:,0], l_hand[:,1], l_hand[:,2], c = 'y')
    ax.scatter(r_hand[:,0], r_hand[:,1], r_hand[:,2], c = 'y')
    ax.scatter(l_knee[:,0], l_knee[:,1], l_knee[:,2], c = 'k')
    ax.scatter(r_knee[:,0], r_knee[:,1], r_knee[:,2], c = 'k')
    ax.scatter(l_feet[:,0], l_feet[:,1], l_feet[:,2], c = 'm')
    ax.scatter(r_feet[:,0], r_feet[:,1], r_feet[:,2], c = 'm')
    #ax.set_zlim(0,2000)
    #ax.set_xlim(0,700)
    #ax.axis('square')
    #ax.axis('off')
    ax.view_init(elev=3,azim=15)
    plt.show(ax)

def _calc_COM_getZ(person,point_list):
    sum = np.zeros([len(person.time_seq)])
    for each_point in point_list:
        #print(person.monitor[each_point][:,2].shape)
        sum += person.monitor[each_point][:,2]
    sum = sum/len(point_list)
    return sum

def calc_COM(person):
    plt.cla()
    COM = 0.0
    para = {}
    para[1] = [[22],[31,32],{"Male":[8.62,46.9],"Female":[8.20,47.3]}]
    para[2] = [[31,32],[29,30],{"Male":[16.82,53.6],"Female":[16.35,49.3]}]
    para[3] = [[29,30],[33],{"Male":[27.23,40.3],"Female":[27.48,44.6]}]
    para[4] = [[13,14],[27,28],{"Male":[14.19,45.3],"Female":[14.10,44.2]}]
    para[5] = [[9,10],[25,26],{"Male":[3.67,39.3],"Female":[4.43,42.5]}]
    para[6] = [[16,17],[18,19],{"Male":[2.43,47.8],"Female":[2.66,46.7]}]
    para[7] = [[18,19],[20,21],{"Male":[1.25,42.4],"Female":[1.14,45.3]}]
    para[8] = [[20,21],[38,39],{"Male":[0.64,36.6],"Female":[0.42,34.9]}]
    para[9] = [[25,26],[34,35],{"Male":[1.48,48.6],"Female":[1.24,45.1]}]
    for each in para:
        ps = para[each][2][person.message[2]][0]/100.0
        ls = para[each][2][person.message[2]][1]/100.0
        eps = 0.9999 if person.message[2] == "Male" else 1.0001
        COM += ps*((1.0-ls)*_calc_COM_getZ(person,para[each][0])+ls*_calc_COM_getZ(person,para[each][1]))*eps
    plt.plot(np.asarray([0.1666667*i for i in range(len(COM))]),COM,'k')
    plt.xlabel("Time")
    plt.ylabel("COM Height")
    plt.grid(linestyle='-.')
    #plt.show()
    save_path = "亚太\\重心高度\\"+person.name+".png"
    plt.savefig(save_path)
    return COM

def calc_feet_height(person):
    plt.cla()
    r_points = [25,5,1,34,3,36]
    l_points = [26,6,2,35,4,37]
    for each_feet in [l_points,r_points]:
        data_sum = np.zeros([len(person.time_seq),3])
        for each_point in each_feet:
            data_seq = person.monitor[each_point]
            data_sum += data_seq
        data_mean = data_sum/6

        feet_height = []
        for index in range(len(data_mean)):
            feet_height.append(data_mean[index][2])

        steps = choose_steplen(feet_height,110,True)
        #if len(steps) > 2:
        #    steps = steps[:-1]
        regr = linear_model.LinearRegression()
        regr.fit(np.reshape(steps,[-1,1]),np.reshape([feet_height[i] for i in steps],[-1,1]))
        regr_coef, regr_intercept = regr.coef_[0][0], regr.intercept_[0]
        #print(regr_coef.shape,regr_intercept.shape)
        #print([0.0166666667*i for i in steps])
        #raise AssertionError()
        for index in range(len(feet_height)):
            feet_height[index] = feet_height[index] - (regr_coef*index+regr_intercept)
        plt.plot(np.asarray([0.01666666667*i for i in range(len(feet_height))]),np.asarray(feet_height))

        single_support = []
        for index in range(len(feet_height)):
            nearest_step = [i for i in steps if i==min([abs(index-t) for t in steps])]
            print([abs(index-t) for t in steps])
            if feet_height[index] < min(feet_height)+3:
                single_support.append(index)
        debug_func=[1 if i in single_support else 0 for i in range(len(person.time_seq))]
        plt.plot(np.asarray([0.01666666667*i for i in range(len(debug_func))]),np.asarray(debug_func))

        leg = "(left leg)" if each_feet == l_points else "(right leg)"
        #print(person.name,leg)

    plt.xlabel("Time/s")
    plt.ylabel("Height/mm")
    plt.grid(linestyle='-.')
    plt.legend(("Left feet","Right feet"))
    plt.show()
    #save_path = "亚太\\步长曲线\\高度曲线_修正\\"+person.name+".png"
    #plt.savefig(save_path)

person_test = Person(file=paths[0])
calc_feet_height(person_test)
#calc_gcenter(person_test)
#gif_gen(person_test)
#calc_COM(person_test)
people = []

#paths = paths[30:35]
total_data_csv = []
for each_p in paths:
    people.append(Person(each_p))
    #calc_feet_height(people[-1])
    #total_data_csv.append(calc_stepsize(people[-1]))
    #gif_gen(people[-1])
print(total_data_csv)

def write_file(data,w_path):
    out = open(w_path,'w',newline='')
    csv_write = csv.writer(out,dialect='excel')
    
    for each_line in data:
        csv_write.writerow([i for i in each_line])

#write_file(total_data_csv,"亚太\\步长.csv")

