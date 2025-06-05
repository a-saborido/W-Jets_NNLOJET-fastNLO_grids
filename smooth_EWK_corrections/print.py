#!/usr/bin/python3

import ROOT

file = ROOT.TFile.Open("EWC.root")
obj = file.Get("QCD/jet_y1_1j")
hist = ROOT.BindObject(obj, ROOT.TH1D)
obj_ew = file.Get("EW/jet_y1_1j")
hist_ew = ROOT.BindObject(obj_ew, ROOT.TH1D)
obj_ewc = file.Get("EWC/jet_y1_1j")
hist_ewc = ROOT.BindObject(obj_ewc, ROOT.TH1D)
#hist_ew.Divide(hist)
hist.Print("all")
hist_ew.Print("all")
hist_ewc.Print("all")
