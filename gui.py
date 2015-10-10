import gtk, glib
import base
import subprocess


class gui (object):

    def __init__(self):
        self.builder = gtk.Builder()
        self.builder.add_from_file("membrain_gui.glade")
        self.builder.connect_signals(self)

    def run(self):
		proc = subprocess.Popen("python2 base.py 1", shell = True,stdout=subprocess.PIPE) # LS ../ IS AN EXAMPLE
		self.builder.get_object("window").show_all()
		glib.io_add_watch(proc.stdout, # FILE DESCRIPTOR
        	glib.IO_IN,  # CONDITION
        	self.write_to_buffer ) # CALLBACK
		gtk.main()

    def write_to_buffer(self, fd, condition):
        if condition == glib.IO_IN: #IF THERE'S SOMETHING INTERESTING TO READ
           char = fd.read(1) # WE READ ONE BYTE PER TIME, TO AVOID BLOCKING
           buf = self.builder.get_object("textview").get_buffer()
           buf.insert_at_cursor(char) # WHEN RUNNING DON'T TOUCH THE TEXTVIEW!!
           return True # FUNDAMENTAL, OTHERWISE THE CALLBACK ISN'T RECALLED
        else:
           return False # RAISED AN ERROR: EXIT AND I DON'T WANT TO SEE YOU ANY
    def window_destroy_cb(self, *args):
        gtk.main_quit()

if __name__ == '__main__':
	#bt = base.bitalino('98:D3:31:80:48:08',1000,[0,1,2,3,4,5])
	gui().run()