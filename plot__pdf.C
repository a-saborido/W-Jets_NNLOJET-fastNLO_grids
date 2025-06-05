// --------------------------------------------------------------
//  plot_pdf.C   — overlay two PDF bands with ratio pad
// --------------------------------------------------------------
#include "Rtypes.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TGraphAsymmErrors.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TString.h"
#include <iostream>

// ------------------------------------------------------------------
void plot_pdf( TString flav  ="g",    double  q2   =1.9,
               TString file1 ="output/plots.root",
               TString flav2 =" ",    TString file2=" " )
{
    // turn off ROOT’s automatic title pads everywhere
    gStyle->SetOptTitle(0);

    // ---------- open FIRST file -----------------------------------
    TFile f1(file1);
    TDirectory* g1 = (TDirectory*) f1.Get("Graphs");
    if (!g1) { std::cout<<"No Graphs dir in "<<file1<<"\n"; return; }

    TString key1;  key1.Form("dir1_q2_%g_pdf_%s", q2, flav.Data());
    TGraphAsymmErrors* p = nullptr;
    g1->GetObject(key1, p);
    if (!p) { std::cout<<"Cannot find "<<key1<<"\n"; return; }
    p->SetTitle("");
    p->SetFillColor (kRed-9);
    p->SetFillStyle (1001);
    p->SetFillColorAlpha(kRed-9, 0.65);
    p->SetLineColor (kRed);
    p->SetLineWidth (1);

    // ---------- optionally open SECOND file -----------------------
    TGraphAsymmErrors* p2 = nullptr;
    if (flav2!=" " && file2!=" ")
    {
        TFile f2(file2);
        TDirectory* g2 = (TDirectory*) f2.Get("Graphs");
        if (!g2){ std::cout<<"No Graphs dir in "<<file2<<"\n"; return; }

        TString key2; key2.Form("dir1_q2_%g_pdf_%s", q2, flav2.Data());
        g2->GetObject(key2, p2);
        if (!p2){ std::cout<<"Cannot find "<<key2<<"\n"; return; }

        p2->SetTitle("");
        p2->SetFillColor (kBlue-9);
        p2->SetFillStyle (3001);
        p2->SetLineColor (kBlue);
        p2->SetLineWidth (1);
    }

    // ---------- build ratio bands --------------------------------
    TGraphAsymmErrors* r  = (TGraphAsymmErrors*) p ->Clone();
    TGraphAsymmErrors* r2 =  p2 ? (TGraphAsymmErrors*) p2->Clone() : nullptr;
    r ->SetTitle("");
    if (r2) r2->SetTitle("");

    for (int i=0;i<p->GetN();++i)
    {
        double val = p->GetY()[i],  err = p->GetErrorY(i);
        double rel = val? err/val : 0.;
        r->SetPoint(i, p->GetX()[i], 1.);
        r->SetPointError(i,0,0,rel,rel);
        if (r2)
        {
            double val1 = p ->GetY()[i];       // central value of reference PDF1
            double val2 = p2->GetY()[i];       // central value of PDF2
            double err2 = p2->GetErrorY(i);    // absolute ±error of PDF2

            double ratio   = val2 / val1;
            double err_abs = err2 / val1;      // error2 divided by central1

            r2->SetPoint      (i, p2->GetX()[i], ratio);
            r2->SetPointError (i, 0, 0, err_abs, err_abs);
        }
    }

    // ---------- canvas & pads ------------------------------------
    const double ratsize = 0.30;           
    const double top_frac = 1.0 - ratsize;  

    // if we want "physically identical" text sizes:
    const double bottom_label = 0.1;
    const double bottom_title = 0.12;
    const double top_label    = bottom_label * (ratsize / top_frac);  
    const double top_title    = bottom_title * (ratsize / top_frac);  

    TCanvas* c=new TCanvas("PDF","pdf",600,600);

    // upper pad
    TPad* pad1=new TPad("pad1","pad1",0.,ratsize,1.,1.);
    pad1->SetLogx();
    pad1->SetBottomMargin(0.02);
    pad1->Draw();
    pad1->cd();

    // Define axis ranges and titles for the main panel
    p->GetXaxis()->Set(101,1e-4,1.);

    // prepend "x " to the Y-axis title
    TString ytitle_main;
    ytitle_main.Form("x %s", flav.Data());
    p->GetYaxis()->SetTitle(ytitle_main);

    // set the top‐pad’s label and title sizes to the *scaled* values
    p->GetYaxis()->SetLabelSize(top_label);
    p->GetYaxis()->SetTitleSize(top_title);
    p->GetYaxis()->SetTitleOffset(0.8);
    p->GetXaxis()->SetLabelSize(0);  // hide bottom‐axis labels in the top pad

    // draw the filled band without error bars ("3"), then the curve on top ("C")
    p->Draw("ACE3");
    if (p2) p2->Draw("CE3 SAME");

    // -------- legend ---------------------------------------------
    TLegend* leg=new TLegend();  // (x1,y1,x2,y2)
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.045);
    // remove "/plots.root" substring from file1 (and file2 if present)
    TString lab1 = file1;
    lab1.ReplaceAll("/plots.root","");
    lab1.ReplaceAll("output_","");
    leg->AddEntry(p , lab1.Data(),"f");
    if (p2) {
        TString lab2 = file2;
        lab2.ReplaceAll("/plots.root","");
        lab2.ReplaceAll("output_","");
        leg->AddEntry(p2, lab2.Data(),"f");
    }
    leg->Draw();

    TLatex tex;
    tex.SetNDC();
    tex.SetTextAlign(34);              // right-top alignment
    tex.SetTextSize(top_label);
    tex.DrawLatex(0.95, 0.92, Form("Q^{2} = %.2f GeV^{2}", q2));

    // lower pad
    c->cd();
    TPad* pad2=new TPad("pad2","pad2",0.,0.,1.,ratsize);
    pad2->SetLogx();
    pad2->SetTopMargin(0.0);
    pad2->SetBottomMargin(0.30);
    pad2->Draw();
    pad2->cd();

    // set up ratio plot axes
    r->SetMaximum(1.35);
    r->SetMinimum(0.65);
    r->GetYaxis()->SetNdivisions(504);

    // add "Ratio" as Y-axis title
    r->GetYaxis()->SetTitle("Ratio");

    // uniform label sizes *for the lower pad*
    r->GetYaxis()->SetLabelSize(bottom_label);
    r->GetYaxis()->SetTitleSize(bottom_title);
    r->GetYaxis()->CenterTitle(kTRUE);
    r->GetYaxis()->SetTitleOffset(0.36);

    r->GetXaxis()->Set(101,1e-4,1.);
    r->GetXaxis()->SetLabelSize(bottom_label);
    r->GetXaxis()->SetTitle("x");
    r->GetXaxis()->SetTitleSize(bottom_title);
    r->GetXaxis()->SetTitleOffset(1.0);

    // draw ratio band with curve
    r->Draw("ACE3");
    if (r2)
    {
        r2->SetFillColor(kBlue-9);
        r2->SetFillStyle(3001);
        r2->SetLineColor(kBlue);
        r2->Draw("CE3");
    }

    c->SaveAs( Form("pdf_%s_Q2_%g.pdf", flav.Data(), q2) );
}
// ------------------------------------------------------------------
