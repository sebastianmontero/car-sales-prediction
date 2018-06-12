
class Utils:
    
    @staticmethod
    def month_id_to_month_of_year(month_id):
        return int(str(int(month_id))[4:])