#!/usr/bin/python3
__author__ = 'James Currier'
__version__ = '3.0'




from PIL import Image, ImageDraw, ImageFont, ExifTags
import numpy
import csv
from random import choice, random
from math import factorial
from os import chdir, listdir, mkdir, path, rename
join = path.join
from statistics import mean, median, stdev
import warnings
import pickle
from time import clock




############################ CHANGE THESE FOR TUNING ##########################

MIN_PLANT_SIZE = 50  # Min number of pixels in a plant
TIGHTNESS = 3  # How tight to trim the rows
SPLITTING_SENSITIVITY = 5/6  # Sensitivity for splitting touching plants
SPEED = max(int(MIN_PLANT_SIZE**0.5 // 2), 1)  # Speed to analysis

###############################################################################




def check_speed():
    t = clock()
    count = 0
    while clock() < t + 0.5:
        factorial(100)
        count += 1

    return count



def distance_between(pt1: tuple, pt2: tuple) -> float:
    """Returns the distance between the two points

    pt1: tuple(int, int)
        point tuple of ints (x, y)
    pt2: tuple(int, int)
        point tuple of ints (x, y)
    """

    return ((pt2[1] - pt1[1])**2 + (pt2[0] - pt1[0])**2)**0.5




def find_local_mins(lis: list, tolerance: float = 0.1) -> list:
    """Finds the local minimums of the list lis
    Returns the indexes of these minimums

    lis: list(number)
        list of numbers to find local minimums of
    """
    
    temp = []
    mins = []

    # Go through the list
    for i in range(len(lis)):

        # get all items below the threshold
        if lis[i] < tolerance:
            temp.append(i)

    # Find the breaks between all of the rows
    curr = temp[0] - 1
    s = temp[0]

    # Step through partial series
    for i in temp:

        # if there has been a break in the series
        if i > curr:

            # Record the middle of the break
            mins.append((s + curr) / 2)

            # Ignore the break and continue following the series
            s = i
            curr = i

        curr += 1

    
    return mins




def within(point: tuple, box: tuple) -> bool:
    """Returns whether point is in box

    point: tuple(int, int)
        point tuple of ints (x, y)
    box: tuple(int, int)
        box tuple of ints (x1, y1, x2, y2)
    """
    
    return box[0] < point[0] < box[2] and box[1] < point[1] < box[3]




def interactive():
    """Interactive use of the plant finder"""

    fn = input('Enter Photo Name or Folder Name: \n')
    print()

    while not path.exists(fn):
        print('{} is not a valid path \n'.format(fn))
        fn = input('Enter Photo Name or Folder Name: \n')
        print()


    # Start Single Mode
    if path.isfile(fn):
        main(fn, MIN_PLANT_SIZE)


    # Start Multiple mode
    else:
        num = input('Enter number of output images: \n')
        print()
        while not num.isdigit():
            print('{} is not a valid number\n'.format(num))
            num = input('Enter number of output images: \n')

        num = int(num)

        options = []
        for i in range(1, num + 1):
            options.append([j.strip()
                            for j in input(
                                'Enter options for output image #{}: \n'
                                .format(i)).split(',')])
            print()
        
        multiple(fn, MIN_PLANT_SIZE, options)




def just_centers(pic_name: str,
                 min_plant_pixels: int = MIN_PLANT_SIZE
                 ) -> None:
    
    """Just finds the plants, doesn't put them into rows.
        Useful for debugging.

    pic_name: str
        Name of the picture to use in this algorithm
    min_plant_pixels
        The smallest number of pixels that constitutes a plant
    """

    # Open Image
    img = Image.open(pic_name)

    print('Time until completion: ' + str(int(round(img.size[0]
                                                    * img.size[1]
                                                    / check_speed()))) + ' seconds')

    # Create a Picture Object
    photo = Picture(img)

    # Create a Field Object without rows    
    global field
    field = photo.find_field(min_plant_pixels, make_rows=False)

    # Show the visual
    field.show_visual()




def just_field(pic_name: str,
               min_plant_pixels: int = MIN_PLANT_SIZE
               ) -> 'Field':
    
    """Main method to start everything

    pic_name: str
        Name of the picture to use in this algorithm
    min_plant_pixels
        The smallest number of pixels that constitutes a plant
    """

    global SPEED
    SPEED = max(int(min_plant_pixels**0.5 // 2), 1)

    # Open image
    img = Image.open(pic_name)

    print('Time until completion: ' + str(int(round(img.size[0]
                                                * img.size[1]
                                                / check_speed()))) + ' seconds')

    # Create a Picture Object
    photo = Picture(img)

    # Create a Field Object with rows
    field = photo.find_field(min_plant_pixels)

    return field




def main(pic_name: str, min_plant_pixels: int = MIN_PLANT_SIZE) -> None:
    """Main method to start everything

    pic_name: str
        Name of the picture to use in this algorithm
    min_plant_pixels
        The smallest number of pixels that constitutes a plant
    """

    global SPEED
    SPEED = max(int(min_plant_pixels**0.5 // 2), 1)

    # Open image
    img = Image.open(pic_name)

    print('Time until completion: ' + str(int(round(img.size[0]
                                                    * img.size[1]
                                                    / check_speed()))) + ' seconds')

    # Create a Picture Object
    photo = Picture(img)

    # Create a Field Object with rows
    global field
    field = photo.find_field(min_plant_pixels)

    # Create a Ruler Object
    global ruler
    ruler = Ruler(field)

    # Measure and save distances
    ruler.output_distances('Distances.csv')
    ruler.output_row_info('Rows.csv')

    print('Location = ' + str(Metadata(img).get_location()) + '\n')
    
    # Show the visual
    field.show_visual(ruler)
 


    
def multiple(folder_name: str,
             min_plant_pixels: int = MIN_PLANT_SIZE,
             output_options = [['rows',
                                'centers',
                                'row_ids',
                                'distances'],
                               
                               ['rows',
                                'centers',
                                'row_ids',
                                'numbers'],
                               
                               ['dirt',
                                'ditches',
                                'rows',
                                'clusters',
                                'centers',
                                'row_ids',
                                'numbers',
                                'lines']
                               ]) -> None:
    
    """Do all of the pictures in folder_name

    pic_name: str
        Name of the picture to use in this algorithm
    min_plant_pixels
        The smallest number of pixels that constitutes a plant
    """

    # Go to the specified folder
    ls = listdir(folder_name)
    ls = [join(folder_name, i) for i in ls]

    # Check if the folder exists
    if join(folder_name, 'Analysis') in ls:

        # If it does, rename the old folder
        new_name = join(folder_name, 'Analysis')
        while new_name in ls:
            new_name += '_old'
            
        rename(join(folder_name,'Analysis'), new_name)


    # Create new folders inside the given directory
    mkdir(join(folder_name, 'Analysis'))
    mkdir(join(folder_name, 'Analysis/Images'))
    mkdir(join(folder_name, 'Analysis/Data'))
    

    # Gather the images to be analysed
    co = 0
    pics = [j for j in ls if path.isfile(j)]
    le = len(pics)

    # Analyse each of the pictures
    for i in pics:

        # Make the field
        field = just_field(i, min_plant_pixels)

        # Measure the field and save results
        print('Saving data...\n')
        ruler = Ruler(field)
        
        ruler.output_distances(
            join(folder_name,
                 'Analysis/Data/{}_Distances.csv'.format(i.split('.')[0])
                 )                 
            )
        
        ruler.output_row_info(
            join(folder_name,
                 'Analysis/Data/{}_Rows.csv'.format(i.split('.')[0])
                 )
            )

        # Make and save visuals
        print('Saving pictures...\n')
        for k in range(len(output_options)):
            output_options[k]
            img = field.make_visual(ruler, output_options[k])
            img.save(
                join(folder_name, 'Analysis/Images/{}_Visual_{}.png'
                .format(i.split('.')[0], k + 1)
                     )
                )

        # Increment the progress meter
        co += 1
        print('Completed {}/{} images\n\n'.format(co, le))




class Picture:
    """A Picture with analytical tools"""
    
    def __init__(self, pic: 'Image') -> None:
        """Picture initializer

        pic: 'Image'
            The image for analysis/ manipulation
        """

        if pic.mode != 'RGB':
            pic = pic.convert('RGB')
            
        self.size = pic.size
        self.photo = pic
        self.pic_array = numpy.array(pic)
        self.bin_pic = self.binarized(self.pic_array)



    def get_array(self) -> numpy.array:
        """Get array version of this Picture"""
        
        return self.pic_array


    
    def get_photo(self) -> 'Image':
        """Photo getter"""
        
        return self.photo



    def get_size(self) -> tuple:
        """Photo size getter"""
        
        return self.size


    def error_occured(self) -> None:
        """Warn the user when an error has occured"""
        
        warnings.warn(
            '''An Error has occured when processing this photo!
               The plants are too emerged in some places to analyze.''',
            RuntimeWarning)


    def binarized(self, pic_array: numpy.ndarray) -> numpy.ndarray:
        """Returns an array of the same size as pic_array, which corresponds
         to whether the pixels in the original array are green.
         
         pic_array: numpy.ndarray
            array to binarize
        """
        
        model = pickle.load(open('GreenNN.bin', 'rb'))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            data = [model[1].predict(model[0].transform(i)) for i in pic_array]

        return numpy.array(data)
        

    def find_cluster(self, point: tuple) -> tuple:
        """Iteratively finds the cluster of green points touching point.
        Returns a tuple containing the bounding box around the cluster and
            a list of the points in the cluster in (x, y) tuples.
        
        point: tuple(x: int, y: int)
            A point in the cluster
        """

        # set quickness
        quickness = 1

        # Create initial bounding box
        bb = [point[0], point[1], point[0], point[1]]

        # Get the first direction to expand the box
        direction = self.touching_border(bb, quickness)


        # loop until the box has no green on the perimeter
        while direction != 'EDGE':

            # Check if there is an error
            if bb[2] - bb[0] > self.size[0] / 4 \
               or bb[3] - bb[1] > self.size[1] / 4:
                
                bb[0] = 0
                bb[2] = len(self.bin_pic[0]) - 1
                return ('ERROR',
                        [(x, y)
                         for x in range(bb[0], bb[2] + 1)
                         for y in range(bb[1], bb[3] + 1)],
                        bb)

            # Expand Down and Right
            if direction == 'BR':
                bb[2] += quickness
                bb[3] += quickness

            # Expand Down and Left
            elif direction == 'BL':
                bb[0] -= quickness
                bb[3] += quickness

            # Expand Right
            elif direction == 'RIGHT':
                bb[2] += quickness

            # Expand Down    
            elif direction == 'BOTTOM':
                bb[3] += quickness

            # Expand Left    
            elif direction == 'LEFT':
                bb[0] -= quickness

            # Expand Up
            elif direction == 'TOP':
                bb[1] -= quickness


            # Check the area directly around the current box
            elif direction == 'NONE':
                cntn = False
                
                for i in range(1, 3):

                    # if there is a green pixel just outside of the box,
                    # expand the box to cover it and continue searching
                    tb = self.touching_border([bb[0] - i,
                                               bb[1] - i,
                                               bb[2] + i,
                                               bb[3] + i])
                    
                    if tb != 'NONE':
                        direction = tb
                        cntn = True
                        break
                    
                if cntn:
                    continue
                break
                

            # Default case
            else:
                raise IndexError(str(direction) + ' is not a valid direction!')


            # Get new direction to expand in
            direction = self.touching_border(bb, quickness)


        # Gather all the green pixels within the bounding box        
        cluster = [(x, y)
                   for x in range(bb[0], bb[2] + 1)
                   for y in range(bb[1], bb[3] + 1)
                   if self.bin_pic[y][x]]


        # Don't count the plant if it's touching the edge of the picture
        if direction == 'EDGE':
            if len(cluster) > 250:
                return (bb, cluster)
            else:
                return (None, cluster, bb)

                    
        return (bb, cluster)


        
    def find_field(self,
                   min_plant_pixels:  float = MIN_PLANT_SIZE,
                   speed: int = SPEED,
                   make_rows: bool = True
                   ) -> 'Field':
        
        """Finds all of the plants in the picture, makes plant objects of them,
            puts them into a Field object and returns them

        min_plant_pixels: int
            The minimum number of green pixels that counts as a plant
        make_rows: bool
            Whether rows should be made
        """
        
        bin_pic_copy = self.bin_pic.copy()

        # Initialize list to return
        plants = []

        # Go through each row of the image
        for y in range(0, len(self.bin_pic), speed):
            if y % 100 == 0:
                print(round(y * 100 / len(self.bin_pic), 1), '% Complete')
            
            # Go through individual pixels of the image
            for x in range(0, len(self.bin_pic[y]), speed):

                # Check if the pixel is green
                if self.bin_pic[y][x]:

                    # Get the cluster of green attached to this pixel
                    cluster = self.find_cluster((x, y))

                    # Check if the plant is on the edge
                    if cluster[0] is None:
                        # Acknowledge that there is a plant here,
                        # but don't count it as part of the field
                        for i in cluster[1]:
                            self.bin_pic[i[1]][i[0]] = 0
                        continue

                    # Check if an error occured
                    if cluster[0] == 'ERROR':
                        # Do not acknoledge the plant, instead,
                        # ignore it, show a warning, and continue
                        for i in cluster[1]:
                            self.bin_pic[i[1]][i[0]] = 0
                        self.error_occured()
                        
                        continue


                    # Check if the 'plant' is actually multiple plants
                    # Make a plant object and find a tight box approximation
                    tb = Plant(cluster[1], cluster[0]).get_tight_box()

                    # Check if the 'plant' is too wide to be a single plant
                    if len(cluster[1]) > 2000 and \
                       (tb[2] - tb[0]) \
                       / (tb[3] - tb[1]) > SPLITTING_SENSITIVITY * 2:

                        # Find the number of plants
                        num = max(2,
                                  int(round(SPLITTING_SENSITIVITY
                                            * (tb[2] - tb[0])
                                            / (tb[3] - tb[1]))))

                        # Sort the plant pixels by x coordinte
                        cluster[1].sort(key=lambda u: u[0])

                        ## Break the plant into num new plants
                        # Initialize variables and find how often to
                        #   chop the plant
                        s = max(1, (cluster[0][2] - cluster[0][0]) / num)
                        news = []
                        x = cluster[0][0]
                        curr = 0

                        # Go through all sections of the current plant
                        while round(x) < cluster[0][2]:
                            x += s
                            curr_old = curr

                            # Skip through in chunks of the size that
                            #   the new plant will be
                            while cluster[1][curr][0] < int(x - 1):
                                curr += 1

                            # make a new cluster of these pixels
                            new_cluster = cluster[1][int(round(curr_old))
                                                     : int(curr)]

                            # Make a new box for each new plant
                            news.append([(min(new_cluster,
                                              key = lambda u: u[0])[0],
                                          min(new_cluster,
                                              key = lambda u: u[1])[1],
                                          max(new_cluster,
                                              key = lambda u: u[0])[0],
                                          max(new_cluster,
                                              key = lambda u: u[1])[1]
                                          ),
                                         new_cluster])
                            

                        # Check if each of the new blobs are big enough
                        # to be considered a plant
                        for i in news:
                            if len(i[1]) >= min_plant_pixels:
                                plants.append(Plant(i[1], i[0]))

                    else:
                        # Check if there are enought pixels in the green
                        # blob to be considered a plant
                        if len(cluster[1]) >= min_plant_pixels:
                            plants.append(Plant(cluster[1], cluster[0]))
                        

                    # Turn the whole green cluster red
                    for i in cluster[1]:
                        self.bin_pic[i[1]][i[0]] = 0


        # After all Plants are found, make a field object
        if make_rows:
            print('\nFinding Rows...\n')
            
        self.bin_pic = bin_pic_copy
        field = Field(self, plants, make_rows)

        return field



    def percent_plant_on_line(self,
                              point: tuple,
                              slope: float,
                              speed: int = SPEED
                              ) -> float:
    
        """Returns percent of green pixels on the
            line created by slope and point

        point: tuple(int, int)
            A point on the desired line
        slope: float
            The slope of the desired line
        speed: int
            The speed at which to check this line
        """
        
        # Initialize variables
        total_green = 0
        total_pixels = 0

        # Go through all x values in the picture
        for i in range(0, self.photo.size[0], speed):

            # Find the y value of the x on the line
            y = int((i - point[0]) * slope + point[1])

            # Count the total green and not green
            if 0 <= y < self.photo.size[1]:
                total_pixels += 1
                if self.bin_pic[y][i]:
                    total_green += 1

        # Return percent of total that are green
        if total_pixels == 0:
            return 0
        
        return total_green / total_pixels



    def touching_border(self, bb: list, quickness: int = 1) -> str:
        """Checks if any of the pixels touching the perimeter
            of bounding box bb are green.
            
        Returns which side has green touching it ('RIGHT',
                                                  'BOTTOM',
                                                  'LEFT',
                                                  'TOP')
        Returns 'NONE' if no green is touching
        Returns 'EDGE' if the bounding box is on the edge of the pic

        bb: list[int, int, int, int] # [x1, y1, x2, y2]
            The bounding box
        quickness: int
            How quickly to search
        """
        
        # Make the coordinates easier to handle
        x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]

        # Check which way the box should be expanded
        # The first two conditions check if the box should
        # be expanded diagonally downwards.
        # These just speed up the box sizing
        
        ## Check if there is a green pixel on the bottom right of the box
        if not (x2 + quickness >= len(self.bin_pic[0])) and \
           not (y2 + quickness >= len(self.bin_pic)):
            if self.bin_pic[y2][x2]:
                return 'BR'

        ## Check if there is a green pixel on the bottom right of the box
        if not (x1 - quickness < 0) and not (y2 + quickness >= len(self.bin_pic)):
            if self.bin_pic[y2][x1]:
                return 'BL'


        # The next four check the sides of the box
        ## Check if there is a green pixel on the right side of the box
        if not (x2 + quickness >= len(self.bin_pic[0])):
            for i in range(y2 - y1 + 1):
                if self.bin_pic[y1 + i][x2]:
                    return 'RIGHT'
                
        ## Check if there is a green pixel on the bottom side of the box
        if not (y2 + quickness >= len(self.bin_pic)):
            for i in range(x2 - x1 + 1):
                if self.bin_pic[y2][x1 + i]:
                    return 'BOTTOM'

        ## Check if there is a green pixel on the left side of the box
        if not (x1 - quickness < 0):
            for i in range(y2 - y1 + 1):
                if self.bin_pic[y1 + i][x1 - quickness]:
                    return 'LEFT'

        ## Check if there is a green pixel on the top side of the box
        if not (y1 - quickness < 0):
            for i in range(x2 - x1 + 1):
                if self.bin_pic[y1 - quickness][x1 + i]:
                    return 'TOP'


        # If all the green pixels are found and the box is on the edge
        if (x2 + quickness >= len(self.bin_pic[0])) or \
           (y2 + quickness >= len(self.bin_pic)) or \
           (y1 - quickness < 0) or \
           (x1 - quickness < 0):
            return 'EDGE'



        # If there are no green pixels touching the box, return 'NONE'
        return 'NONE'




class Plant:
    """A Plant Object that contains information about each plant"""

    def __init__(self, cluster: list, box: tuple) -> None:
        """Plant Initializer

        cluster: list(tuple(int, int))
            list of (x, y) coordinates where parts of this plant are
        box: tuple(int, int, int, int
            (x1, y1, x2, y2) tuple of the box around this plant
        """
        
        self.cluster = cluster
        self.box = box
        self.center = self.find_center()
        self.make_tight_box()



    def get_box(self) -> tuple:
        """Box getter"""
        
        return self.box



    def get_center(self) -> tuple:
        """Center getter"""
        
        return self.center



    def get_cluster(self) -> list:
        """Cluster getter"""
        
        return self.cluster


    
    def get_diameter(self) -> float:
        """Gets approximate diameter of this plant"""
        
        return (self.box[3] - self.box[1] + self.box[2]  - self.box[0]) / 2



    def get_tight_box(self) -> tuple:
        """tight_box Getter"""
       
        return self.tight_box
    


    def set_box(self, box: tuple) -> None:
        """Box setter

        box: tuple(int, int, int, int)
            tuple (x1, y1, x2, y2) around the plant
        """
        
        self.box = box


    
    def set_cluster(self, cluster: list) -> None:
        """Cluster setter

        cluster: list(tuple(int, int))
            list of (x, y) coordinates where this plant is
        """

        self.cluster = cluster
        


    def find_center(self) -> tuple:
        """Finds the center of points
        Returns (x, y) tuple of center pixel

        points: list(tuple(x: int, y: int))
            The points to find the center of
        """
        
        # Add up all the x values of pixels in the plant
        # Then divide by total pixels in the plant
        avg_x = sum([i[0] for i in self.cluster]) / len(self.cluster)

        # Add up all the y values of pixels in the plant
        # Then divide by total pixels in the plant
        avg_y = sum([i[1] for i in self.cluster]) / len(self.cluster)

        self.center = (int(round(avg_x)), int(round(avg_y)))
        
        # return the results in a tuple of integers
        return self.center



    def make_tight_box(self, tightness: float = 0.33) -> None:
        """Finds a box that more tightly approximates the plant.
        Sets this plant's tight_box to this new box.

        tightness: float [0, 1]
            How tight to make the box
            Represents the amount of green on each side of the box
        """

        # Default to the plant's original box
        x1 = self.box[0]
        y1 = self.box[1]
        x2 = self.box[2]
        y2 = self.box[3]


        ## Find y coordinates
        # Initialize variables and sort pixels by x coordinate
        width = self.box[2] - self.box[0]
        cents = sorted(self.cluster, key = lambda u: u[1])
        clust = []
        curr = cents[0][1]
        last = 0

        # Split the pixels by x coordinate
        for p in range(len(cents)):
            if cents[p][1] != curr:
                clust.append(cents[last: p])
                curr = cents[p][1]
                last = p
                
        # Get the topmost y value which is <tightness> green
        for hor in clust:
            if len(hor) / width > tightness:
                y1 = hor[0][1]
                break

        # Get the bottommost y value which is <tightness> green
        for hor in clust[::-1]:
            if len(hor) / width  > tightness:
                y2 = hor[0][1]
                break


        ## Find x coordinates
        # Initialize variables and sort pixels by y coordinate
        height = self.box[3] - self.box[1]
        cents = sorted(self.cluster, key = lambda u: u[0])
        clust = []
        curr = cents[0][0]
        last = 0

        # Split the pixels by y coordinate
        for p in range(len(cents)):
            if cents[p][0] != curr:
                clust.append(cents[last: p])
                curr = cents[p][0]
                last = p

        # Get the leftmost x value which is <tightness> green
        for ver in clust:
            if len(ver) / height > tightness:
                x1 = ver[0][0]
                break

        # Get the rightmost x value which is <tightness> green
        for ver in clust[::-1]:
            if len(ver) / height > tightness:
                x2 = ver[0][0]
                break

        # Default to original x values if no better estimate was found
        if x1 == x2:
            x1 = self.box[0]
            x2 = self.box[2]

            
        # Default to original y values if no better estimate was found
        if y1 == y2:
            y1 = self.box[1]
            y2 = self.box[3]


        self.tight_box = (x1, y1, x2, y2)




class Field:
    """Field object that contains Plants and
        has tools for analyzing its Plants"""
    
    def __init__(self,
                 picture: 'Picture',
                 plants: list,
                 make_rows: bool = True
                 ) -> None:
        
        """Field Initializer

        picture: 'Picture'
            The picture of the field
        plants: list('Plant')
            list of plants in the field
        make_rows: bool
            whether rows should be made
        """
        
        self.picture = picture
        self.plants = plants
        self.centers = [i.get_center() for i in plants]
        
        if make_rows:
            self.find_field_angle()
            self.make_rows()
            self.remove_outliers(TIGHTNESS)



    def get_average_plant_width(self, sample_size: int = 1000) -> float:
        """Find and return the average plant width of sample_size

        sample_size: int
            The number of plants to sample
        """
        
        total = 0

        # The sample size can not be larger than the field size
        if sample_size > self.get_num_plants():
            sample_size = self.get_num_plants()


        # Pick sample_size random plants and sum their diameters
        total = 0
        
        for _ in range(sample_size):
            p = choice(self.plants)
            total += p.get_diameter()


        # Get the average plant diameter
        avg = total / sample_size
        
        return avg


    
    def get_boxes(self) -> list:
        """Boxes getter"""
        
        return [i.get_box() for i in self.plants]



    def get_centers(self) -> list:
        """Centers getter"""
        
        return self.centers



    def get_ditches(self) -> list:
        """Ditches getter"""

        return self.ditches


    
    def get_clusters(self) -> list:
        """Clusters getter"""
        
        return [i.get_cluster() for i in self.plants]



    def get_lines(self) -> list:
        """Lines getter"""
        
        return self.lines



    def get_num_plants(self) -> int:
        """Returns number of plant in this Field"""
        
        return len(self.plants)



    def get_picture(self) -> 'Picture':
        """Pic getter"""
        
        return self.picture



    def get_plants(self) -> list:
        """Plants getter"""
        return self.plants



    def get_row_spacing(self) -> tuple:
        """Return the median distance between rows"""

        # Get the y coordinate of the line through each of the
        # field's rows at the middle of the image
        ys = [line[0] * (self.get_picture().get_size()[0] / 2) + line[1]
              for line in self.lines]

        # Get the distances between each of the approximated rows and sort them
        dists = sorted([ys[i] - ys[i - 1] for i in range(1, len(ys))])

        # Take the median of the distances
        dist_px = dists[len(dists)//2]

    
        return dist_px



    def get_rows(self) -> list:
        """Rows getter"""
        
        return self.rows



    def get_slope(self) -> float:
        """Slope getter"""
        
        return self.slope



    def set_slope(self, slope: float) -> None:
        """Set the slope of the rows in this field

        slope: float
            The new slope
        """
        
        self.slope = slope
            


    def get_tight_boxes(self) -> list:
        """Tight Boxes getter"""

        return [plant.get_tight_box() for plant in self.plants]


        
    def add_plant(self, plant: 'Plant') -> None:
        """Add a plant to this field

        plant: 'Plant'
            The plant to add
        """

        self.plants.append(plant)
        self.centers.append(plant.get_center())
        self.make_rows()



    def remove_plant_by_center(self, center: tuple) -> None:
        """Remove plants from the field by thier center point

        center: tuple
            the center to be removed
        """

        # Remove the center from the list of plant centers
        self.centers.remove(center)

        # Remove the plant with this center from the list of plants
        for i in range(len(self.plants)):
            if self.plants[i].get_center() == center:
                self.plants.pop(i)
                break

        # Remove the plant with this center from the list of rows
        for i in self.rows:
            for j in range(len(i)):
                if i[j].get_center() == center:
                    i.pop(j)
                    break
            else:
                continue
            break



    def find_field_angle(self) -> None:
        """Find and set the angle of this Field"""

        # Gather required info
        size = self.picture.get_size()
        
        # Make a center box in the field
        # Make sure there are no more than 50 plants in this box
        smol = 5
        while len([i
                   for i in self.plants
                   if within(i.get_center(), (smol//2*size[0]//smol,
                                              smol//2*size[1]//smol,
                                              (smol//2+1)*size[0]//smol,
                                              (smol//2+1)*size[1]//smol))]
                  ) > 50:
            
            smol += 1

        small_box = (2*size[0]//smol,
                     2*size[1]//smol,
                     3*size[0]//smol,
                     3*size[1]//smol)

        slopes = []

        # iterate throught the centers of all plants in the box 
        for pivot in [i for i in self.centers if within(i, small_box)]:

            # Get a pivot point within the small_box
            pivot = (-1, -1)
            while not within(pivot, small_box):
                pivot = choice(self.centers)

            # Initialize variables
            best_slope = 0
            percent_green = 0

            # Go throught all of the plants within the small_box
            for i in self.centers:
                if within(i, small_box) and i != pivot:

                    # Determine how much green a line draw
                    # from the pivot to the new point hits
                    try:
                        slope = (pivot[1] - i[1]) / (pivot[0] - i[0])
                    except ZeroDivisionError:
                        continue
                    PGOL = self.picture.percent_plant_on_line(i, slope)

                    # If the green hit is more than the current green,
                    # replace it
                    if PGOL > percent_green:
                        best_slope = slope
                        percent_green = PGOL

            # Save the best slope
            slopes.append(best_slope)

        # Find and set the median of all the best slopes
        self.slope = median(slopes)



    def make_rows(self, tolerance: float = 0.05, speed: int = SPEED) -> None:
        """Finds and sets the rows of this Field

        torerance: float
            The amount of plant in a line that is considered to be a row
        speed: int
            The speed at which to find the rows
        """

        PGOLS = []

        # Find the y values for which a line of slope self.slope
        # through (0, y) is on the top and bottom of the picture
        top_bound = int(round(min(0, self.slope
                                  * self.picture.get_size()[0]
                                  * -1)))
        
        bottom_bound = int(round(max(self.picture.get_size()[1],
                                     self.picture.get_size()[0]
                                     * -1
                                     * self.slope
                                     + self.picture.get_size()[1])))

        # Go though the entire picture in lines of slope self.slope
        for i in range(top_bound, bottom_bound, speed):

            # Record the percent of this line that is a plant
            for _ in range(speed):
                PGOLS.append(self.picture.percent_plant_on_line((0, i),
                                                               self.slope))
        PGOLS.append(0)
        
        # Find the local mins in the amounts of plant on the lines
        mins = find_local_mins(PGOLS, tolerance)

        # Adjust for where the measurements began
        for i in range(len(mins)):
            mins[i] += top_bound

        # Get the distances between each of the breaks
        dist = []
        
        for i in range(1, len(mins)):
            dist.append(mins[i] - mins[i-1])

        # Get the median of these distances
        avg = median(dist)

        # Remove breaks that are too close to eachother.
        curr = 2
        while curr < len(mins):
            if (mins[curr] - mins[curr - 1]) < avg * 0.5 \
               and mins[curr - 1] >= 1:
                
                mins.pop(curr)
                
            else:
                curr += 1
        
        mins.append(bottom_bound)

        
        # Initialize a dictionary with keys that correspond to the local mins
        rows = {}
        for i in mins:
            rows[i] = []

        # Determine between which local mins each plant lies 
        for plant in self.plants:

            # Find where a line of slope self.slope through the center
            # of this plant would land on the y axis        
            i = plant.get_center()
            b = i[1] - self.slope * i[0]

            # Separate the plants by these values
            curr = 0
            while b > mins[curr]:
                curr += 1
                
            rows[mins[curr]].append(plant)


        # Use the dictionary to create rows in the field
        self.rows = [rows[i] for i in rows if len(rows[i]) >= 3]

        # Sort the field's rows
        self.sort_rows()

        # Make a line of best fit for each of the rows 
        self.lines = [numpy.polyfit([plant.get_center()[0]
                                     for plant in row],
                                    [plant.get_center()[1]
                                     for plant in row], 1)
                      for row in self.rows]

        # Make a line for each of the ditches
        self.ditches = [(self.slope, val) for val in mins]



    def remove_outliers(self, tolerance: int = 2):
        """Cuts out just the rows, ignoring the ditches between them.
        This is good for cutting out weeds and
        random debris between rows.

        tolerance: int
            How close to trim the rows
        """
        global row_dist

        # Find the median distance between the rows of the field in pixels
        d = []
        for i in range(1, len(self.lines)):
            d.append(abs(self.lines[i][1] - self.lines[i-1][1]))
            
        row_dist = median(d)

        # Iterate through all of the rows
        for row_num in range(len(self.rows)):
            i = 0

            # Iterate through each plant in the row
            while i < len(self.rows[row_num]):

                # Find each plants distance from it's line
                c = self.rows[row_num][i].get_center()
                dis = abs(c[0]
                          * self.lines[row_num][0]
                          + self.lines[row_num][1]
                          - c[1])

                # If it's too far off, remove it
                if dis < row_dist/tolerance:
                    i += 1
                else:
                    self.remove_plant_by_center(c)
                    


    def sort_rows(self) -> list:
        """Uses insertion sort to sort rows by x coordinate.
        Mutates list

        rows: list
            list of rows in the form of lists of tuples (x, y)
        """

        # Sort Inner rows
        done_rows = []

        # Sort each row independently
        for r in self.rows:
            
            # Insertion Sort by x coordinate
            r = r[::-1] # reverse the list first to speed up sorting
                        # this is faster because the list is
                        # already partly reverse sorted
            
            sorted_row = []
            
            while r != []:

                # Take element from row
                el = r.pop()

                # Find where it belongs
                curr = 0
                while curr < len(sorted_row) and \
                      el.get_center()[0] > sorted_row[curr].get_center()[0]:
                    
                    curr += 1

                # Insert it there
                sorted_row.insert(curr, el)

            # collect sorted rows
            done_rows.append(sorted_row)


        # Sort Inter rows
        finished = []
        
        while done_rows != []:

            # Take out a row
            el = done_rows.pop()

            # Find where it belongs
            curr = 0
            while curr < len(finished) and \
                  (finished[curr][0].get_center()[1]
                   + finished[curr][len(finished[curr]) - 1].get_center()[1]) \
                   / 2 < (el[0].get_center()[1]
                          + el[len(el) - 1].get_center()[1]) / 2:
                curr += 1

            # Insert it there
            finished.insert(curr, el)


        # Set the sorted rows
        self.rows = finished



    def make_visual(self,
                    ruler = None,
                    options: list = ['rows',
                                     'centers',
                                     'distances']
                    ) -> 'Image':
        
        """Creates a visual view of the analysed field

        ruler: 'Ruler'
            The ruler whose distances to use
        options: list
            Which layers to display
        """
        
        original = self.get_picture().get_photo()

        # Copy the original image for drawing
        img = original.copy()
        draw = ImageDraw.Draw(img)

        # check all the choices provided by the user
        for i in options:
            
            if i == 'clusters':
                # Color all cluster pixels red
                
                for j in self.get_clusters():
                    for k in j:
                        img.putpixel(k, (25,275,25))


            elif i == 'row_ids':
                # Make row id numbers

                # Font specifications
                size = 75
                font = ImageFont.truetype('ariblk.ttf', size)
                color = (88, 214, 216)
                num = 1

                # Draw the ids
                for j in self.rows:
                    draw.text((j[0].get_center()[0],
                               j[0].get_center()[1] - 0.25 * size),
                              str(num),
                              fill = color,
                              font = font)
                    num += 1


            elif i == 'boxes':
                # Show all bounding boxes
                
                for i in self.get_boxes():
                    draw.rectangle(i, outline=(255, 0, 255))


            elif i == 'dirt':
                # Remove Background
                
                img = Image.new('RGB', img.size, (130, 90, 50))
                draw = ImageDraw.Draw(img)


            elif i == 'centers':
                # Show all centers
                
                rad = 9
                for i in self.get_centers():
                    draw.arc([(i[0] - rad, i[1] - rad),
                              (i[0] + rad, i[1] + rad)],
                             0, 360, (0, 0, 255))


            elif i == 'ditches':
                # Show ditches between plants

                # Line attribute settings
                width = 10
                color = (55,65,65)

                # Iterate over all ditches
                for line in self.ditches:
                    line = [line[0], line[1]]

                    # Point in ditch on left border of picture
                    start_point = (0, line[1])

                    # Point in ditch on right border of picture
                    end_point = (self.picture.get_size()[0] - 1,
                                 line[0]
                                 * (self.picture.get_size()[0] - 1)
                                 + line[1])


                    ## Check if the end point is within the picture
                    if end_point[1] < 0:

                        if start_point[1] < 0:
                            continue
                        
                        # Point in ditch on top border of picture
                            end_point = (-1 * line[1] / line[0], 0)


                    elif end_point[1] > self.picture.get_size()[1] - 1:

                        if start_point[1] > self.picture.get_size()[1] - 1:
                            continue
                        
                        # Point in ditch on bottom border of picture
                        end_point = (-1
                                     * (self.picture.get_size()[1] - 1)
                                     / line[0],
                                     self.picture.get_size()[1] - 1)

                    # Draw the ditches
                    for i in self.get_rows():
                        draw.line((start_point, end_point), color, width)


            elif i == 'lines':
                # Show row line approximations

                # Line attribute settings
                width = 1
                color = (255, 255, 75)

                # Iterate over all the lines
                for line in self.lines:
                    line = [line[0], line[1]]

                    # Point on line on left border of picture
                    start_point = (0, line[1])

                    # Point on line on right border of picture
                    end_point = (self.picture.get_size()[0] - 1,
                                 line[0]
                                 * (self.picture.get_size()[0] - 1)
                                 + line[1])


                    ## Check if the end point is within the picture
                    if end_point[1] < 0:

                        if start_point[1] < 0:
                            continue

                        # Point on line on top border of picture
                            end_point = (-1 * line[1] / line[0], 0)
                            

                    elif end_point[1] > self.picture.get_size()[1] - 1:

                        if start_point[1] > self.picture.get_size()[1] - 1:
                            continue

                        # Point on line on bottom border of picture
                        end_point = (-1
                                     * (self.picture.get_size()[1] - 1)
                                     / line[0],
                                     self.picture.get_size()[1] - 1)


                    # Draw the lines
                    for i in self.get_rows():
                        draw.line((start_point, end_point), color, width)


            elif i == 'rows':
                if self.get_rows():
                    # Show lines between rows
                    
                    width = 3
                    color = (255,0,0)

                    for i in self.get_rows():
                        draw.line([j.get_center() for j in i], color, width)
                else:
                    print('Rows have not been made for this field')


            elif i == 'numbers':
                # Display numbers between plants

                # Find where to put the numbers
                midpoints = [(int(round((row[c].get_center()[0]
                                         + row[c + 1].get_center()[0]) / 2)),
                              int(round((row[c].get_center()[1]
                                         + row[c + 1].get_center()[1]) / 2)))
                             
                             for row in self.get_rows()
                             for c in range(len(row) - 1)]

                # Font specifications
                size = 10
                font = ImageFont.truetype('ariblk.ttf', size)
                num = 1

                # Write numbers
                for i in midpoints:
                    draw.text((i[0] - 3 * len(str(round(num, 1))),
                               i[1]),
                              str(round(num,1)), font = font)
                    
                    num += 1


            elif i == 'tight':
                # Display tight boxes

                for i in self.get_tight_boxes():
                    draw.rectangle(i, outline=(100, 255, 255))


            elif i == 'distances':
                # display distances between plants

                # find where to put the distances
                midpoints = [(int(round((row[c].get_center()[0]
                                         + row[c + 1].get_center()[0]) / 2)),
                              int(round((row[c].get_center()[1]
                                         + row[c + 1].get_center()[1]) / 2)))
                             for row in self.get_rows()
                             for c in range(len(row) - 1)]

                # Font specifications
                size = 10
                font = ImageFont.truetype('arial.ttf', size)
                num = 1

                # Write numbers
                for i in midpoints:
                    draw.text((i[0] - 3 * len(str(ruler.get_distances()[num])),
                               i[1]),
                              str(ruler.get_distances()[num]) + '"',
                              font = font)
                    
                    num += 1


            # If the user inputs something that isn't an option    
            else:
                raise Exception(i + ' is not a valid option.\n')

            
        return img

        

    def show_visual(self, ruler: 'Ruler') -> None:
        
        ## Used for Debugging / Display
        
        """Interactive function that shows a visual view of
            what the algorithm has done
        Asks for user input on type of photo show, and shows it

        ruler: 'Ruler':
            The ruler whose distances to use
        """

        # Mainloop of the program
        show = True
        
        while show: 
            imgshow = True
            imgsave = False

            # Get the user's choice
            choice = input(
                'Which layer to display? (Type "help" for options)\n'
                )
            
            print()
            
            choice = choice.split(',')

            options = []

            # Check if the user inputted a special option
            for i in choice:
                i = i.strip().lower()

                if i == 'quit':
                    # quit
                    
                    show = False
                    break


                elif i == 'help':
                    # print help, and restart
                    
                    print('''The options are:
                                              default
                                              digital
                                              ----------
                                              boxes
                                              clusters
                                              centers
                                              dirt
                                              distances
                                              ditches
                                              lines
                                              numbers
                                              row_ids
                                              rows
                                              tight
                                              ----------
                                              save

                             Separate multiple options with commas.
                             Enter "quit" to exit.\n''')
                    
                    choice = []
                    break


                elif i == 'save':
                    # Save analysed image
                    
                    imgsave = True


                elif i == '' or i == 'default':
                    # Default view
                    
                    options.extend(['rows',
                                    'centers',
                                    'distances',
                                    'row_ids'])


                elif i == 'digital':
                    # Show what the algorithm sees

                    options.extend(['dirt',
                                    'rows',
                                    'clusters',
                                    'boxes',
                                    'centers',
                                    'ditches',
                                    'lines'])
                    
                
                elif i in ['boxes',
                           'clusters',
                           'centers',
                           'distances',
                           'numbers',
                           'rows',
                           'ditches',
                           'lines',
                           'tight',
                           'dirt',
                           'row_ids']:
                    
                    options.append(i)

                else:
                    print('{} is not a vaild option!\n'.format(i))
                    break
                
            else:      

                # Make analysed image
                img = self.make_visual(ruler, options)
                if img is None:
                    continue

                # Show the image if appropriate
                if imgshow:
                    img.show()

                # Save the image if appropriate
                if imgsave:
                    img.save('LastSave.png')




class Ruler:
    """Measuring tools for field objects"""

    def __init__(self, field: 'Field') -> None:
        """Ruler Initializer

        field: 'Field'
            The Field object to measure
        """
        
        self.field = field
        self.ratio = self.get_distance_scale(36)
        self.photo_name = field.get_picture().get_photo().filename
        self.location = Metadata(
            field.get_picture().get_photo()
            ).get_location()
        
        self.find_distances_and_rows()



    def get_distance_scale(self, row_spacing: float) -> tuple:
        """Assume that the average distance between the
            rows is row_spacing inches.
        Use this to return a tuple of (px, inches) ratio

        row_spacing: float
            distance between rows in inches
        """
    
        return (self.field.get_row_spacing(), row_spacing)



    def get_field(self) -> 'Field':
        """Field getter"""
        
        return self.field



    def get_location(self) -> tuple:
        """Location getter"""

        return self.location


        
    def get_photo_name(self) -> str:
        """Photo Name getter"""

        return self.photo_name
    


    def set_ratio(self, ratio: tuple) -> None:
        """Sets the ruler's pixels to inches ratio

        ratio: tuple
            A ratio of pixels to inches
        """
        
        self.ratio = ratio



    def get_distances(self) -> dict:
        """Distances getter"""
        
        return self.dists



    def find_distances_and_rows(self) -> None:
        """Find the distances between plants in this field
            and find the row which each gap belongs to
        """
        
        # Initialize results dictionary and counter
        dists = {}
        rows = {}
        num = 1
        curr_row = 1

        # Add all the results to a dictionary with unique ids
        for row in self.field.get_rows():
            for c in range(len(row) - 1):

                # Get distance between plant and the one next to it
                d = distance_between(row[c].get_center(),
                                     row[c + 1].get_center())
                
                d = self.pixels_to_inches(d)
                d = round(d, 2)

                # Add the distance to the results dictionary
                dists[num] = d
                rows[num] = curr_row
                num += 1

            curr_row += 1

        # Set this rulers dists and rows
        self.dists = dists
        self.rows = rows



    def output_distances(self, output_file: str) -> None:
        """Gets the distances between all of the plants in a row.
        Outputs all distances to a csv file.
    
        output_file: str
            The name of the file to write distances to
        """
        
        # Open the given file for writing
        with open(output_file, 'w+', newline = '') as csvfile:
            out = csv.writer(csvfile)

            # Make header
            out.writerow(['Gap ID',
                          'Distance (Inches)',
                          'Row ID', 'Latitude',
                          'Longitude',
                          'Image Name'])

            # write each of the rows to the file
            for i in self.dists:
                out.writerow([i,
                              self.dists[i],
                              self.rows[i],
                              self.location[0],
                              self.location[1],
                              self.photo_name])
        
        print('Distances written to ' + output_file + '\n')



    def output_row_info(self, output_file: str) -> None:
        """Gets information about the rows.
        Outputs all information to a csv file.
        
        output_file: str
            The name of the file to write distances to
        """

        with open(output_file, 'w+', newline = '') as csvfile:
            out = csv.writer(csvfile)

            # Make header
            out.writerow(['Row ID',
                          'Ideal Number of Plants',
                          'Actual Number of Plants',
                          'Average Gap (Inches)',
                          'Median Gap (Inches)',
                          'Standard Deviation (Inches)',
                          'Latitude',
                          'Longitude',
                          'Image Name'])
            
            # Write information about each row to the file
            row_id = 0
        
            for row in self.field.get_rows():

                # Calculate the length of this row
                row_dist = self.pixels_to_inches(
                    distance_between(row[0].get_center(),
                                     row[-1].get_center()))

                rds = [self.pixels_to_inches(
                    distance_between(
                        row[i].get_center(), row[i+1].get_center()))
                       for i in range(len(row) - 1)]


                # Calculate statistics to write               
                row_id += 1
                ideal_num = row_dist // 16 + 1
                act_num = len(row)
                avg_gap = mean(rds)
                median_gap = median(rds)
                std_dev = stdev(rds)
                   

                # Write results to file
                out.writerow([row_id,
                              ideal_num,
                              act_num,
                              avg_gap,
                              median_gap,
                              std_dev,
                              self.location[0],
                              self.location[1],
                              self.photo_name])
                 
            
        print('Row information written to ' + output_file + '\n')
            


    def pixels_to_inches(self, px: float) -> float:
        """Converts pixels to inches
        Returns a float of inches

        px: float
            The number of pixels to convert
        """
        
        return (px / self.ratio[0]) * self.ratio[1]




class Metadata:
    """Metadata from a jpg"""
    
    def __init__(self, jpg: 'Image') -> None:
        """Metadata Initializer

        jpg: 'Image'
            The image to remove metadata from
        """

        # Get raw exif from the image
        try:
            info = jpg._getexif()
            self.data = {}
        
            # Check if there is any exif data
            if info == None:
                return None

            # Match exif labels to english keys
            for i in info:
                if i not in ExifTags.TAGS.keys():
                    continue

                # Handle GPS data separately
                if i != 34853:
                    self.data[ExifTags.TAGS[i]] = info[i]
                    
                else:
                    self.data[ExifTags.TAGS[i]] = self._get_GPS_data(info[i])


        except AttributeError:
            info = None
            self.data = None



    def _get_GPS_data(self, gpsdata: dict) -> dict:
        """Convert exif metadata to english

        gpsdata: dict
            The exif gps data from the jpg
        """
        
        ans = {}

        # Replace all exif tags with english titles
        for i in gpsdata:
            ans[ExifTags.GPSTAGS[i]] = gpsdata[i]
            
        return ans



    def get_data(self) -> dict:
        """Data Getter"""
        
        return self.data



    def get_location(self) -> tuple:
        """Location Getter
            Returns a tuple containing (latitude, longitude)
        """

        if self.data is None:
            return (None, None)
        
        lat = self.data['GPSInfo']['GPSLatitude']
        lon = self.data['GPSInfo']['GPSLongitude']
        
        # Convert from Degrees, minutes, seconds to standard form
        latitude = (lat[0][0] / lat[0][1]) \
                   + (lat[1][0] / lat[1][1] / 60) \
                   + (lat[2][0] / lat[2][1] / 3600)
        
        longitude = (lon[0][0] / lon[0][1]) \
                    + (lon[1][0] / lon[1][1] / 60) \
                    + (lon[2][0] / lon[2][1] / 3600)


        # Adjust for direction references
        if self.data['GPSInfo']['GPSLatitudeRef'] == 'S':
            latitude *= -1

        if self.data['GPSInfo']['GPSLongitudeRef'] == 'W':
            longitude *= -1
            

        return (round(latitude, 6), round(longitude, 6))



    def get_location_str(self) -> tuple:
        """Location Getter
            Returns a tuple containing (latitude, longitude)
        """

        if self.data is None:
            return (None, None)

        lat = self.data['GPSInfo']['GPSLatitude']
        lon = self.data['GPSInfo']['GPSLongitude']
        
        # Convert from Degrees, minutes, seconds to standard form
        latitude = (lat[0][0] / lat[0][1]) \
                   + (lat[1][0] /lat[1][1] / 60) \
                   + (lat[2][0] / lat[2][1] / 3600)
        
        longitude = (lon[0][0] / lon[0][1]) \
                    + (lon[1][0] / lon[1][1] / 60) \
                    + (lon[2][0] / lon[2][1] / 3600)
        

        # Make the results presentable
        latitude = str(round(latitude, 6)) \
                   + chr(176) + ' ' \
                   + self.data['GPSInfo']['GPSLatitudeRef']
        
        longitude = str(round(longitude, 6)) \
                    + chr(176) + ' ' \
                    + self.data['GPSInfo']['GPSLongitudeRef']

        
        return (latitude, longitude)




if __name__ == '__main__':
    interactive()
