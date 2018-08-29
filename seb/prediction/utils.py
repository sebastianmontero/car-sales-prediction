import os
import glob
import re

class Utils:
    
    @staticmethod
    def month_id_to_month_of_year(month_id):
        return int(str(int(month_id))[4:])
    
    @staticmethod
    def add_months_to_month_id(month_id, num_months):
        month_id = str(int(month_id))
        year = int(month_id[:4])
        month = int(month_id[4:])
        year += (month + num_months - 1) // 12
        month = ((month + num_months -1) % 12) + 1 
        return Utils.get_month_id(year, month)
    
    @staticmethod
    def get_month_id(year, month):
        month_id = str(year)
        if month < 10:
            month_id +='0'
        month_id += str(month) 
        return int(month_id)

    @staticmethod
    def remove_files_from_dir(path, prefixes):
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                for prefix in prefixes:
                    if file.startswith(prefix):
                        os.remove(os.path.join(path,file))
                      
    @staticmethod
    def escape_brackets(str_):
        new_str = ''
        for c in str_:
            if c == '[':
                new_str += '[[]'
            elif c == ']':
                new_str += '[]]'
            else:
                new_str += c
        return new_str 
    
    @staticmethod
    def filter_list(list_, filter_=None, exclude_filter=None):        
        if filter_:
            list_ = [p for p in list_ if re.search(filter_,p)]
        
        if exclude_filter:
            list_ = [p for p in list_ if re.search(exclude_filter,p) is None]
            
        return list_

    '''
        recursive can be boolean or an int if int it indicates the number of subdirs to explore under
        the current path
    ''' 
    @staticmethod
    def search_paths(base_path, path_end, recursive=False, sort=False, filter_=None, exclude_filter=None):
        
        max_subdirs = 5
        
        base_path = Utils.escape_brackets(base_path)
        if isinstance(recursive, bool):    
            path_wild_card = '**' if recursive else ''
            path = os.path.join(base_path, path_wild_card, path_end)
            paths = glob.glob(path, recursive=recursive)
        else:
            paths = []
            path_wild_card = ''
            for i in range(max_subdirs):
                path = os.path.join(base_path, path_wild_card, path_end)
                tpaths = glob.glob(path)
                paths += tpaths
                if isinstance(recursive, int):
                    if i >= recursive:
                        break;
                else:
                    if len(tpaths) > 0 and tpaths[0].find(recursive) >= 0:
                        break;
                path_wild_card +='*/'
        
        paths = Utils.filter_list(paths, filter_, exclude_filter)
            
        if sort:
            paths.sort(reverse=(sort == 'reverse'))
        return paths