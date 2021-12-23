import schedule
import time
import threading

from naver_news import update_news

def run_continuously(interval=1):
    
    cease_continuous_run = threading.Event()

    class ScheduleThread(threading.Thread):
        @classmethod
        def run(cls):
            while not cease_continuous_run.is_set():
                schedule.run_pending()
                time.sleep(interval)

    continuous_thread = ScheduleThread()
    continuous_thread.start()
    return cease_continuous_run

def start_news_update_scheduler():
    # 8:23 9:00 10:1 11:2 12:3 1:4 2:5 3:6
    for hour in ['23', '00', '01', '02', '03', '04', '05', '06']:
        schedule.every().monday.at(f"{hour}:00").do(update_news)
        schedule.every().tuesday.at(f"{hour}:00").do(update_news)
        schedule.every().wednesday.at(f"{hour}:00").do(update_news)
        schedule.every().thursday.at(f"{hour}:00").do(update_news)
        schedule.every().friday.at(f"{hour}:00").do(update_news)


    # Start the background thread
    stop_run_continuously = run_continuously()

    # # Do some other things...
    # time.sleep(120)


    # # Stop the background thread
    # stop_run_continuously.set()