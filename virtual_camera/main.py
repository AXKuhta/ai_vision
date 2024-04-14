from tkinter import *
import numpy as np

##########################################################################################
# Rendering + display
##########################################################################################

params = {
	"near": 1,
	"far": 10,
	"k1": 0.0,
	"transform_matrix":  np.array([
		[1., 0., 0., 0.],
		[0.,-1., 0., 0.],
		[0., 0., 1.,-5.],
		[0., 0., 0., 1.]
	]),
	"projection_matrix":  np.array([ # https://observablehq.com/@esperanc/model-view-and-projection-demo
		[1., 0., 0., 0.],
		[0., 1., 0., 0.],
		[0., 0., 1., 0.],
		[0., 0.,-1., 1.],
	])
}

root = Tk()

w = 720
h = 720

canvas = Canvas(root, width=w, height=h)
canvas.pack()

canvas.create_line(0, 0, 200, 100)
canvas.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))

canvas.create_rectangle(50, 25, 150, 75, fill="blue")

pts = np.loadtxt("cube-full.txt")
pts = np.loadtxt("head.txt")
pad = np.ones([pts.shape[0], 1])
pts = np.hstack([pts, pad])

def update_matrix(near, far):
	params["projection_matrix"][2][2] = (near+far)/(near-far)
	params["projection_matrix"][2][3] = (2*near*far)/(near-far)

update_matrix(1, 10)

def render():
	canvas.delete("all")

	tpts = (params["projection_matrix"] @ params["transform_matrix"] @ pts.T).T

	for pt in tpts:
		x, y, z, _w = pt

#		# Near/far clip
#		if z < 0 or z > 1:
#			continue

		# Barrel/pincushion
		r = (x**2 + y**2)**0.5
		x = x * (1 + params["k1"]*r**2)
		y = y * (1 + params["k1"]*r**2)

		x = x/_w * w/2 + w/2
		y = y/_w * h/2 + h/2
		z = z/_w

		print(x, y, z)

		canvas.create_oval(x, y, x, y, fill="black")


render()

##########################################################################################
# UI
##########################################################################################

controls_frame = Frame(root, bg="cyan", pady=3, padx=3)
controls_frame.pack(side="right")

grid_frame = Frame(controls_frame, pady=3, padx=3)
grid_frame.grid(row=0, column=0)

def mk_entry():
	return Entry(grid_frame, bd=2 , width=5, justify="center")

# Matrix inp
inputs = [
	[ mk_entry(),  mk_entry(),  mk_entry(),  mk_entry() ],
	[ mk_entry(),  mk_entry(),  mk_entry(),  mk_entry() ],
	[ mk_entry(),  mk_entry(),  mk_entry(),  mk_entry() ],
	[ mk_entry(),  mk_entry(),  mk_entry(),  mk_entry() ],
]

for i, a in enumerate(inputs):
	for j, b in enumerate(a):
		b.grid(row=i, column=j)

def update_pincushion(x):
	params["k1"] = float(x)
	render()

pincushion_frame = Frame(controls_frame)
pincushion_frame.grid(row=0, column=1)

pincushion = Scale(pincushion_frame, orient="horizontal", label="K1", length=150, from_=-0.03, to=0.03, resolution=0.001, tickinterval=0.03, command=update_pincushion)
pincushion.grid()

def update_translation_x(x):
	params["transform_matrix"][0][3] = float(x)
	render()

def update_translation_y(x):
	params["transform_matrix"][1][3] = float(x)
	render()

translation_frame = Frame(controls_frame)
translation_frame.grid(row=0, column=2)

translation_x = Scale(translation_frame, orient="horizontal", label="X", length=150, from_=-2, to=2, resolution=0.1, tickinterval=1, command=update_translation_x)
translation_y = Scale(translation_frame, orient="horizontal", label="Y", length=150, from_=-2, to=2, resolution=0.1, tickinterval=1, command=update_translation_y)
translation_x.grid()
translation_y.grid()

#apply = Button(controls_frame, text="Render", command=render)
#apply.grid(row=0, column=2)

"""
# •Entry()

def ECallBack():  L.configure(text=E.get())

EnB= Button(top_frame,text="EntryButton",command= ECallBack )
E  .grid(row=0, column=1)
E  .focus_set()
EnB.grid(row=0, column=2)
"""

#mainloop()
