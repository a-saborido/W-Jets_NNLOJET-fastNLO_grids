#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#include "TFile.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TGraphErrors.h"
#include "TGraphSmooth.h"

// ------------------------------------------------------------------
// 1) manual 5-bin weighted smoother (unchanged)
// ------------------------------------------------------------------
TH1D* SmoothHistogramWithErrors(const TH1D* h)
{
   const int nBins = h->GetNbinsX();
   auto* h_smoothed = (TH1D*)h->Clone("h_manual");
   h_smoothed->Reset();

   for (int i = 1; i <= nBins; ++i) {
      double sumWeightedContent = 0., sumWeights = 0., sumErr2 = 0.;
      for (int j = i - 2; j <= i + 2; ++j) {
         if (j < 1 || j > nBins) continue;
         const double c = h->GetBinContent(j);
         const double e = h->GetBinError  (j);
         const double w = h->GetBinWidth  (j);
         if (e <= 0. || w <= 0.) continue;
         const double weight = 1.0/(e*e*w);
         sumWeightedContent += weight * c;
         sumWeights         += weight;
         sumErr2            += weight * weight * e * e;
      }
      if (sumWeights > 0.) {
         h_smoothed->SetBinContent(i, sumWeightedContent / sumWeights);
         h_smoothed->SetBinError  (i, std::sqrt(sumErr2) / sumWeights);
      }
   }
   return h_smoothed;
}

// ------------------------------------------------------------------
// 2) statistically-weighted Super-Smoother
// ------------------------------------------------------------------
TH1D* SmoothHistogramSuper(const TH1D* h,
                           double bass = 0.,  // smoothing “frequency” (0 = auto)
                           double span = 0.)  // span (0 = CV choice)
{
   const int nBins = h->GetNbinsX();

   // a) wrap the histogram in a TGraphErrors
   auto* g = new TGraphErrors(nBins);
   std::vector<double> weights(nBins);

   for (int i = 1; i <= nBins; ++i) {
      const double x  = h->GetBinCenter (i);
      const double y  = h->GetBinContent(i);
      const double dy = h->GetBinError  (i);
      g->SetPoint       (i-1, x, y);
      g->SetPointError  (i-1, 0., dy);            // σY = dy
      weights[i-1] = (dy > 0.) ? 1.0/(dy*dy) : 0.; // w = 1/σ²
   }

   // b) run Super-Smoother (TGraphSmooth keeps ownership!)
   TGraphSmooth gs;
   TGraph* gSmooth = gs.SmoothSuper(g, "", bass, span, kFALSE, weights.data());

   // c) pour the smoothed values back into a TH1
   auto* hSmooth = (TH1D*)h->Clone("h_super");
   hSmooth->Reset("ICE");                   // keep axes & stats

   for (int i = 0; i < nBins; ++i) {
      double xx, yy;
      gSmooth->GetPoint(i, xx, yy);
      const int bin = i + 1;               // TH1 bins start at 1
      hSmooth->SetBinContent(bin, yy);
      hSmooth->SetBinError  (bin, 0.);     // or whatever you prefer
   }

   delete g;          // safe: only the *input* graph
   return hSmooth;    // gSmooth will be deleted by gs’ destructor
}


// ------------------------------------------------------------------
// 3) main macro
// ------------------------------------------------------------------
void smooth_EWC()
{
   gStyle->SetOptStat(0);

   // 1) open file & fetch raw histogram
   const std::string histName = "W_pt_1j_Wp"; //jet_y1_1j_Wp, jet_pt1_1j_Wp, HT_1j_Wp, W_pt_1j_Wp
   const std::string histPath = "EWC/" + histName;    

   TFile *f = TFile::Open("EWC.root");
   if (!f || f->IsZombie()) { printf("Cannot open EWC.root\n"); return; }

   TH1D *h_raw = nullptr;
   f->GetObject(histPath.c_str(), h_raw);             // use path variable
   if (!h_raw) { printf("Histogram %s not found.\n", histPath.c_str()); return; }

   // 2) produce smoothed versions
   TH1D *h_manual = SmoothHistogramWithErrors(h_raw);          // 5-bin window
   TH1D *h_kernel = (TH1D*)h_raw->Clone("h_kernel");           // built-in
   h_kernel->Smooth(10, "width");                              // 10× K5 kernel
   TH1D *h_super  = SmoothHistogramSuper(h_raw);               // Super-Smoother

   // 3) style
   h_raw   ->SetMarkerStyle(20); h_raw   ->SetMarkerSize(0.8);
   h_manual->SetMarkerStyle(24); h_manual->SetMarkerSize(0.8);
   h_kernel->SetMarkerStyle(25); h_kernel->SetMarkerSize(0.8);
   h_super ->SetMarkerStyle(22); h_super ->SetMarkerSize(0.8);

   h_raw   ->SetLineColor(kBlack);   h_raw   ->SetMarkerColor(kBlack);
   h_manual->SetLineColor(kBlue+1);  h_manual->SetMarkerColor(kBlue+1);
   h_kernel->SetLineColor(kRed+1);   h_kernel->SetMarkerColor(kRed+1);
   h_super ->SetLineColor(kGreen+2); h_super ->SetMarkerColor(kGreen+2);

   // 4) draw 
   auto *c = new TCanvas("c","RAW vs smoothed",1200,850);  
   c->SetGrid();
   c->SetLeftMargin  (0.12);  
   c->SetRightMargin (0.05);
   c->SetBottomMargin(0.13);  
   c->SetTopMargin   (0.05);

   h_raw->SetTitle("");
   h_raw->GetXaxis()->SetTitleOffset(1.1);  
   h_raw->GetYaxis()->SetTitleOffset(1.4); 


   h_raw->GetXaxis()->SetTitleSize(0.045);   
   h_raw->GetYaxis()->SetTitleSize(0.045);   
   h_raw->GetXaxis()->SetLabelSize(0.040);   
   h_raw->GetYaxis()->SetLabelSize(0.040);  
   
   // --------------------------------------------------------------
   // Fix Y-axis limits (example values)
   // --------------------------------------------------------------
   const double YMIN = 0.2;      // lower edge you want
   const double YMAX = 1.4;    // upper edge you want
   h_raw->SetMinimum(YMIN);
   h_raw->SetMaximum(YMAX);
   
   h_raw   ->Draw("E1");
   h_manual->Draw("E1 SAME");
   h_kernel->Draw("E1 SAME");
   h_super ->Draw("E1 SAME");

   auto *leg = new TLegend(0.60,0.70,0.88,0.88);
   leg->SetTextSize(0.032);
   leg->AddEntry(h_raw,    "raw data"              ,"lep");
   leg->AddEntry(h_manual, "manual 5-bin weighted" ,"lep");
   leg->AddEntry(h_kernel, "TH1::Smooth(10,\"width\")","lep");
   leg->AddEntry(h_super,  "Super-Smoother (1/\\sigma^{2})","lep");
   leg->SetBorderSize(0);
   leg->Draw();

   const std::string pdfName = histName + "_smoothed.pdf";
   c->SaveAs(pdfName.c_str());

   /* ------------------------------------------------------------
    * 5) dump the smoothed contents
    *    three columns:  bin-low  bin-high  value
    * ---------------------------------------------------------- */
   auto dump = [](const char* tag, const TH1D* h)
   {
      const int n = h->GetNbinsX();
      printf("\n%s\n%-14s %-14s %-14s\n",
             tag,"low","high","content");
      for (int i = 1; i <= n; ++i) {
         const double low  = h->GetBinLowEdge(i);
         const double high = low + h->GetBinWidth(i);
         const double val  = h->GetBinContent(i);
         printf("%14.6g %14.6g %14.6g\n", low, high, val);
      }
   };

   dump("Manual 5-bin smoother",  h_manual);
   dump("TH1::Smooth(10,\"width\")", h_kernel);
   dump("Super-Smoother (1/σ²)",  h_super);
} 

