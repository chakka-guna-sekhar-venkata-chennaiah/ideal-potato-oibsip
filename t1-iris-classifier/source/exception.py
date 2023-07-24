

import sys
from source.loggers import logging

def error_message_detail(error,error_detail=sys):
    _,_,error_tf=error_detail.exc_info()
    file_name=error_tf.tb_frame.f_code.co_filename
    file_no=error_tf.tb_lineno
    error_message='Error occured in the python script name {} line number {} error message {}'.format(
        file_name,file_no,str(error))

    return error_message


class custom_exception(Exception):
    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    
if __name__=='__main__':
    try:
        a=1/2
        a=a/0
        print(a)
    except Exception as e:
        #print(e)
        logging.info('Divide by error zero')
        raise custom_exception(e,sys)