import os

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